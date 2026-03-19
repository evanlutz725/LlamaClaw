from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.clients import BraveSearchClient, OllamaClient
from app.config import Settings
from app.models import ChatMessage, CommandDecision, ResearchPlan
from app.repositories import ConversationRepository, MemoryRepository, OnboardingRepository, UserProfileRepository
from app.services import ContextAssembler, MemoryRefreshWorker, OnboardingService
from app.storage import JsonFileStore

LOGGER = logging.getLogger(__name__)


class LlamaClawBot:
    def __init__(
        self,
        settings: Settings,
        conversation_repo: ConversationRepository,
        memory_repo: MemoryRepository,
        profile_repo: UserProfileRepository,
        onboarding_repo: OnboardingRepository,
        context_assembler: ContextAssembler,
        ollama_client: OllamaClient,
        brave_client: BraveSearchClient,
        refresh_worker: MemoryRefreshWorker,
    ) -> None:
        self.settings = settings
        self.conversation_repo = conversation_repo
        self.memory_repo = memory_repo
        self.profile_repo = profile_repo
        self.onboarding_repo = onboarding_repo
        self.context_assembler = context_assembler
        self.ollama_client = ollama_client
        self.brave_client = brave_client
        self.refresh_worker = refresh_worker

        self.application = Application.builder().token(settings.telegram_bot_token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("clearcontext", self.clear_context))
        self.application.add_handler(CommandHandler("resetcontext", self.clear_context))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        chat_id = str(update.effective_chat.id)
        profile = self.profile_repo.load(chat_id)
        onboarding_state = self.onboarding_repo.load(chat_id)
        if not profile.onboarding_complete:
            onboarding_state.started = True
            self.onboarding_repo.save(onboarding_state)
            await update.message.reply_text(OnboardingService.first_prompt())
            return
        await update.message.reply_text(
            "LlamaClaw is online. Send a message to research something, or use /clearcontext to wipe only the rolling chat context."
        )

    async def clear_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return
        chat_id = str(update.effective_chat.id)
        self.conversation_repo.save(type(self.conversation_repo.load(chat_id))(chat_id=chat_id))
        self.refresh_worker.reset_chat_context(chat_id)
        await update.message.reply_text(
            "Conversation context cleared for this chat. Your onboarding profile and long-term learned memory were kept."
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat:
            return

        chat_id = str(update.effective_chat.id)
        user_text = update.message.text or ""
        profile = self.profile_repo.load(chat_id)
        onboarding_state = self.onboarding_repo.load(chat_id)

        if not profile.onboarding_complete:
            if onboarding_state.completed:
                onboarding_state.started = False
                onboarding_state.current_step = 0
                onboarding_state.completed = False
                onboarding_state.answers = {}
            if not onboarding_state.started:
                onboarding_state.started = True
                self.onboarding_repo.save(onboarding_state)
                await update.message.reply_text(OnboardingService.first_prompt())
                return
            updated_profile, updated_state, next_question = OnboardingService.record_answer(onboarding_state, profile, user_text)
            self.profile_repo.save(updated_profile)
            self.onboarding_repo.save(updated_state)
            if next_question:
                await update.message.reply_text(next_question)
            else:
                await update.message.reply_text(
                    "Onboarding complete. I now know your main goals and preferences, and I'll use them in future replies."
                )
            return

        conversation = self.conversation_repo.load(chat_id)
        conversation.messages.append(ChatMessage(id=str(uuid.uuid4()), role="user", text=user_text))
        self.conversation_repo.save(conversation)
        self.refresh_worker.record_message_and_maybe_refresh(chat_id)

        command = await self._decide_command(user_text, profile, conversation.messages)
        research_context = None
        search_results = []
        crawled_pages = []
        if command.cmd == "research":
            research_plan = await self._build_research_plan(user_text, command, profile, conversation.messages)
            aggregated = await self._run_research_workers(research_plan)
            search_results = aggregated["search_results"]
            crawled_pages = aggregated["crawled_pages"]
            research_context = (
                "Internet access is available for this reply through a multi-worker research pipeline.\n"
                "The app spawned independent research workers, saved their outputs to temp files, and aggregated them here.\n"
                "Use the following research packet as your browsing context.\n"
                "When site crawl pages are present, synthesize across the whole crawled site rather than treating it as a single page.\n"
                "Legitimate business prospecting and public lead research are allowed tasks.\n"
                "Be explicit that you are using web research when relevant.\n\n"
                f"Command decision: {command.model_dump_json(exclude_none=True)}\n"
                f"Research plan: {research_plan.model_dump_json(exclude_none=True)}\n\n"
                + aggregated["context"]
            )

        memory = self.memory_repo.load()
        prompt_messages = self.context_assembler.build_messages(
            memory,
            conversation.messages,
            profile=profile,
            research_context=research_context,
        )
        response_text = await self._generate_response(prompt_messages, search_results)
        response_text = self._append_sources(response_text, search_results)

        conversation.messages.append(ChatMessage(id=str(uuid.uuid4()), role="assistant", text=response_text))
        self.conversation_repo.save(conversation)
        self.refresh_worker.record_message_and_maybe_refresh(chat_id)
        await update.message.reply_text(response_text)

    async def _decide_command(self, text: str, profile, conversation: list[ChatMessage]) -> CommandDecision:
        direct_url = self._extract_url(text)
        profile_summary = self.context_assembler._build_profile_summary(profile) if profile else ""
        recent_context = "\n".join(f"{message.role}: {message.text}" for message in conversation[-6:])
        planner_messages = [
            {
                "role": "system",
                "content": (
                    "You classify the user's next requested action for an assistant. "
                    "Return strict JSON only with keys cmd, search, url, reason. "
                    "cmd must be either chat or research. "
                    "Use research when the user is asking for facts, investigation, online discovery, external validation, or to look into a person, brand, site, or company. "
                    "Treat business prospecting, market mapping, competitor research, and public lead generation as normal research tasks. "
                    "Set search to the best concise search query. "
                    "Set url if the user gave a direct URL or clearly wants a specific website crawled. "
                    "Use the user's profile and recent conversation to resolve references like 'my business', 'what now', or 'that company'. "
                    "Do not choose research for vague planning or reflective follow-ups unless external information is actually needed. "
                    "Do not include markdown fences or extra commentary."
                ),
            },
            {
                "role": "system",
                "content": (
                    "User profile context:\n"
                    + (profile_summary or "- No completed onboarding profile yet.")
                    + "\n\nRecent conversation:\n"
                    + (recent_context or "- No recent messages.")
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            raw = await self.ollama_client.chat(planner_messages)
            data = self._extract_json_object(raw)
            return CommandDecision.model_validate(data)
        except Exception:
            if direct_url:
                return CommandDecision(cmd="research", search=urlparse(direct_url).netloc, url=direct_url, reason="fallback_url_detected")
            lower = text.lower().strip()
            if lower.startswith("research:") or lower.startswith("/research") or bool(
                re.search(r"\b(search|research|look up|find online|investigate|see what you find)\b", lower)
            ):
                return CommandDecision(cmd="research", search=self._research_query(text), reason="fallback_keyword_detected")
            return CommandDecision(cmd="chat", reason="fallback_chat")

    async def _build_research_plan(self, text: str, command: CommandDecision, profile, conversation: list[ChatMessage]) -> ResearchPlan:
        profile_summary = self.context_assembler._build_profile_summary(profile) if profile else ""
        recent_context = "\n".join(f"{message.role}: {message.text}" for message in conversation[-6:])
        planner_messages = [
            {
                "role": "system",
                "content": (
                    "You create a research fan-out plan for a local research bot. "
                    "Return strict JSON only with keys search_queries, crawl_urls, goal. "
                    "Generate multiple focused search queries that cover different angles of the task. "
                    "Prefer 6 to 12 search queries. "
                    "If a specific site should be crawled, include it in crawl_urls. "
                    "Use the user profile to resolve references like my business or my company. "
                    "Do not include markdown fences or commentary."
                ),
            },
            {
                "role": "system",
                "content": (
                    "User profile context:\n"
                    + (profile_summary or "- No completed onboarding profile yet.")
                    + "\n\nRecent conversation:\n"
                    + (recent_context or "- No recent messages.")
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            raw = await self.ollama_client.chat(planner_messages)
            data = self._extract_json_object(raw)
            plan = ResearchPlan.model_validate(data)
            if command.search and command.search not in plan.search_queries:
                plan.search_queries.insert(0, command.search)
            if command.url and command.url not in plan.crawl_urls:
                plan.crawl_urls.insert(0, command.url)
            return plan
        except Exception:
            base_query = command.search or self._research_query(text)
            return ResearchPlan(
                goal=base_query,
                search_queries=self._fallback_research_queries(base_query, profile),
                crawl_urls=[command.url] if command.url else [],
            )

    @staticmethod
    def _research_query(text: str) -> str:
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text.strip()

    @staticmethod
    def _extract_url(text: str) -> str | None:
        match = re.search(r"(https?://[^\s]+)", text)
        return match.group(1) if match else None

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in planner response")
        return json.loads(match.group(0))

    @staticmethod
    def _select_crawl_target(search_results: list) -> str | None:
        return search_results[0].url if search_results else None

    def _fallback_research_queries(self, base_query: str, profile) -> list[str]:
        queries = [
            base_query,
            f"{base_query} official website",
            f"{base_query} services",
            f"{base_query} team founder",
            f"{base_query} pricing offer",
            f"{base_query} reviews case study",
            f"{base_query} linkedin",
            f"{base_query} contact",
        ]
        if profile and profile.active_projects:
            queries.extend(f"{project} {base_query}" for project in profile.active_projects[:2])
        deduped: list[str] = []
        for query in queries:
            if query and query not in deduped:
                deduped.append(query)
        return deduped[: self.settings.research_parallel_search_workers]

    async def _run_research_workers(self, plan: ResearchPlan) -> dict:
        work_dir = Path(tempfile.mkdtemp(prefix="llamaclaw-research-"))
        search_queries = plan.search_queries[: self.settings.research_parallel_search_workers]
        crawl_urls = plan.crawl_urls[: self.settings.research_parallel_crawl_workers]
        tasks = []
        for index, query in enumerate(search_queries):
            output_path = work_dir / f"search_{index}.json"
            tasks.append(
                self._spawn_worker(
                    mode="search",
                    output_path=output_path,
                    query=query,
                )
            )
        for index, url in enumerate(crawl_urls):
            output_path = work_dir / f"crawl_{index}.json"
            tasks.append(
                self._spawn_worker(
                    mode="crawl",
                    output_path=output_path,
                    url=url,
                )
            )

        if not tasks and plan.goal:
            output_path = work_dir / "search_0.json"
            tasks.append(self._spawn_worker(mode="search", output_path=output_path, query=plan.goal))

        output_paths = await asyncio.gather(*tasks)
        aggregated_summaries: list[str] = []
        search_results = []
        crawled_pages = []

        for output_path in output_paths:
            if not output_path.exists():
                continue
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            summary = payload.get("summary", "")
            if summary:
                aggregated_summaries.append(f"{payload.get('worker_type', 'worker')}::{payload.get('target', '')}\n{summary}")
            if payload.get("worker_type") == "search":
                search_results.extend(payload.get("results", []))
            elif payload.get("worker_type") == "crawl":
                crawled_pages.extend(payload.get("pages", []))

        context = (
            "Worker outputs:\n"
            + "\n\n".join(aggregated_summaries)
            + ("\n\nNo worker output captured." if not aggregated_summaries else "")
        )
        return {
            "context": context,
            "search_results": search_results,
            "crawled_pages": crawled_pages,
        }

    async def _spawn_worker(self, mode: str, output_path: Path, query: str | None = None, url: str | None = None) -> Path:
        command = [
            sys.executable,
            "-m",
            "app.research_worker",
            mode,
            "--api-key",
            self.settings.brave_api_key,
            "--output",
            str(output_path),
            "--count",
            str(self.settings.brave_search_results),
            "--scrape-count",
            str(self.settings.brave_scrape_results),
            "--max-pages",
            str(self.settings.site_crawl_max_pages),
            "--max-chars",
            str(self.settings.site_crawl_chars_per_page),
        ]
        if query:
            command.extend(["--query", query])
        if url:
            command.extend(["--url", url])

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()
        return output_path

    @staticmethod
    def _append_sources(response_text: str, search_results: list) -> str:
        if not search_results:
            return response_text
        source_lines = ["", "Sources:"]
        for result in search_results[:5]:
            if isinstance(result, dict):
                source_lines.append(f"- {result.get('title', 'Untitled')}: {result.get('url', '')}")
            else:
                source_lines.append(f"- {result.title}: {result.url}")
        return response_text.strip() + "\n" + "\n".join(source_lines)

    async def _generate_response(self, prompt_messages: list[dict[str, str]], search_results: list) -> str:
        draft = await self.ollama_client.chat(prompt_messages)
        if not self.settings.self_review_enabled:
            return draft

        review_messages = [
            *prompt_messages,
            {"role": "assistant", "content": draft},
            {
                "role": "system",
                "content": (
                    "Review the assistant draft carefully before sending it. "
                    "Look for factual errors, unsupported claims, weak reasoning, and missing caveats. "
                    "If research results were supplied, do not state anything that is not grounded in them. "
                    "Do not over-refuse legitimate business research, public prospecting, or public-company lead generation tasks. "
                    "Prefer a useful answer using public information over a generic safety refusal. "
                    "Rewrite the answer so it is slower, more careful, and more accurate. "
                    "Return only the improved final answer."
                ),
            },
            {"role": "user", "content": "Produce the best final answer now."},
        ]
        return await self.ollama_client.chat(review_messages)


def ensure_default_system_prompt(settings: Settings, store: JsonFileStore) -> str:
    return store.read_text(
        settings.system_prompt_path,
        default=(
            "You are LlamaClaw, a careful Telegram chatbot and research assistant. "
            "You have access to persisted local conversation files and a unified long-term memory file. "
            "You can access the internet through Brave Search and site crawling when research results are supplied by the app. "
            "Use the supplied memory, onboarding profile, and recent conversation context. "
            "Think step by step internally before answering. "
            "Prefer accuracy over speed, say when you are unsure, and avoid confident guesses. "
            "Treat legitimate business research, prospecting, market mapping, and public lead generation as normal allowed work. "
            "Prefer public business information and public contact channels when helping with outreach or lead discovery. "
            "When research results are present, ground your response in them, acknowledge that you used web research, and cite sources plainly."
        ),
    )
