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
from app.models import ChatMessage, ChatSettings, CommandDecision, ResearchOutline, ResearchPlan
from app.repositories import (
    ChatSettingsRepository,
    ConversationRepository,
    MemoryRepository,
    OnboardingRepository,
    UserProfileRepository,
)
from app.services import ContextAssembler, MemoryRefreshWorker, OnboardingService
from app.storage import JsonFileStore

LOGGER = logging.getLogger(__name__)
TELEGRAM_MESSAGE_LIMIT = 3900


class LlamaClawBot:
    def __init__(
        self,
        settings: Settings,
        conversation_repo: ConversationRepository,
        memory_repo: MemoryRepository,
        profile_repo: UserProfileRepository,
        onboarding_repo: OnboardingRepository,
        chat_settings_repo: ChatSettingsRepository,
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
        self.chat_settings_repo = chat_settings_repo
        self.context_assembler = context_assembler
        self.ollama_client = ollama_client
        self.brave_client = brave_client
        self.refresh_worker = refresh_worker

        self.application = Application.builder().token(settings.telegram_bot_token).build()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("enable", self.enable_debug))
        self.application.add_handler(CommandHandler("disable", self.disable_debug))
        self.application.add_handler(CommandHandler("deepthinking", self.enable_deepthinking))
        self.application.add_handler(CommandHandler("normalthinking", self.disable_deepthinking))
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

    async def enable_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_chat_mode(update, debug_enabled=True, reply="Debug tracing enabled. I will post structured research traces and prompt previews in this chat.")

    async def disable_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_chat_mode(update, debug_enabled=False, reply="Debug tracing disabled.")

    async def enable_deepthinking(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_chat_mode(
            update,
            deep_thinking_enabled=True,
            reply="Deep thinking enabled. I will use much larger context, broader fan-out, and more aggressive research synthesis for this chat.",
        )

    async def disable_deepthinking(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._set_chat_mode(update, deep_thinking_enabled=False, reply="Deep thinking disabled. Returning to the standard research profile.")

    async def _set_chat_mode(
        self,
        update: Update,
        debug_enabled: bool | None = None,
        deep_thinking_enabled: bool | None = None,
        reply: str = "",
    ) -> None:
        if not update.message or not update.effective_chat:
            return
        chat_id = str(update.effective_chat.id)
        chat_settings = self.chat_settings_repo.load(chat_id)
        if debug_enabled is not None:
            chat_settings.debug_enabled = debug_enabled
        if deep_thinking_enabled is not None:
            chat_settings.deep_thinking_enabled = deep_thinking_enabled
        self.chat_settings_repo.save(chat_settings)
        await update.message.reply_text(reply)

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
        chat_settings = self.chat_settings_repo.load(chat_id)

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
        if chat_settings.debug_enabled:
            await self._emit_debug(update, "command", {
                "latest_user_request": user_text,
                "command": command.model_dump(exclude_none=True),
            })
        research_context = None
        search_results = []
        crawled_pages = []
        if command.cmd == "research":
            research_plan = await self._build_research_plan(user_text, command, profile, conversation.messages)
            if chat_settings.debug_enabled:
                await self._emit_debug(update, "research_plan", research_plan.model_dump(exclude_none=True))
            aggregated = await self._run_research_workers(research_plan, chat_settings)
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
            if chat_settings.debug_enabled:
                await self._emit_debug(update, "aggregated_research", {
                    "worker_count": aggregated["worker_count"],
                    "search_result_count": len(search_results),
                    "crawl_page_count": len(crawled_pages),
                    "context_preview": aggregated["context"][:3000],
                })

        memory = self.memory_repo.load()
        chat_window_size = self.settings.chat_window_size * (4 if chat_settings.deep_thinking_enabled else 1)
        prompt_messages = self.context_assembler.build_messages(
            memory,
            conversation.messages[-chat_window_size:],
            profile=profile,
            research_context=research_context,
        )
        if chat_settings.debug_enabled:
            await self._emit_debug(update, "prompt_preview", self._summarize_prompt(prompt_messages))
        if command.cmd == "research":
            response_text = await self._generate_research_report(
                prompt_messages,
                user_text,
                chat_settings,
                update,
            )
        else:
            response_text = await self._generate_response(prompt_messages, search_results, chat_settings, update)
        response_text = self._append_sources(response_text, search_results)

        conversation.messages.append(ChatMessage(id=str(uuid.uuid4()), role="assistant", text=response_text))
        self.conversation_repo.save(conversation)
        self.refresh_worker.record_message_and_maybe_refresh(chat_id)
        await self._reply_in_chunks(update, response_text)

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

    async def _run_research_workers(self, plan: ResearchPlan, chat_settings: ChatSettings) -> dict:
        work_dir = Path(tempfile.mkdtemp(prefix="llamaclaw-research-"))
        search_limit = self.settings.research_parallel_search_workers * (3 if chat_settings.deep_thinking_enabled else 1)
        crawl_limit = self.settings.research_parallel_crawl_workers * (3 if chat_settings.deep_thinking_enabled else 1)
        search_queries = plan.search_queries[: search_limit]
        crawl_urls = plan.crawl_urls[: crawl_limit]
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
            "worker_count": len(output_paths),
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

    async def _generate_response(
        self,
        prompt_messages: list[dict[str, str]],
        search_results: list,
        chat_settings: ChatSettings,
        update: Update | None = None,
    ) -> str:
        draft = await self.ollama_client.chat(prompt_messages)
        if chat_settings.debug_enabled and update is not None:
            await self._emit_debug(update, "draft", {"draft_preview": draft[:2500]})
        if not self.settings.self_review_enabled:
            return draft
        latest_user_request = next(
            (message["content"] for message in reversed(prompt_messages) if message["role"] == "user"),
            "",
        )

        review_messages = [
            *prompt_messages,
            {"role": "assistant", "content": draft},
            {
                "role": "system",
                "content": (
                    "Review the assistant draft carefully before sending it. "
                    "Your job is to answer the user's latest request directly, not to comment on missing context. "
                    "If the draft is weak, confused, or meta, rewrite it from scratch using the available research and conversation context. "
                    "Look for factual errors, unsupported claims, weak reasoning, and missing caveats. "
                    "If research results were supplied, do not state anything that is not grounded in them. "
                    "Do not over-refuse legitimate business research, public prospecting, or public-company lead generation tasks. "
                    "Prefer a useful answer using public information over a generic safety refusal. "
                    "Never say you were not given a prompt, question, or context if a user request exists. "
                    "Rewrite the answer so it is slower, more careful, and more accurate. "
                    "Return only the improved final answer."
                ),
            },
            {"role": "user", "content": f"Produce the best final answer to this request now: {latest_user_request}"},
        ]
        if chat_settings.debug_enabled and update is not None:
            await self._emit_debug(update, "review_prompt_preview", self._summarize_prompt(review_messages))
        return await self.ollama_client.chat(review_messages)

    async def _generate_research_report(
        self,
        prompt_messages: list[dict[str, str]],
        latest_user_request: str,
        chat_settings: ChatSettings,
        update: Update | None = None,
    ) -> str:
        outline = await self._build_research_outline(prompt_messages, latest_user_request)
        if chat_settings.debug_enabled and update is not None:
            await self._emit_debug(update, "research_outline", outline.model_dump())

        report_dir = Path(tempfile.mkdtemp(prefix="llamaclaw-report-"))
        report_path = report_dir / "report.txt"
        report_parts = [self._plain_text_heading(outline.title)]
        report_path.write_text(report_parts[0] + "\n\n", encoding="utf-8")

        for index, section in enumerate(outline.sections, start=1):
            section_text = await self._generate_report_section(
                prompt_messages,
                latest_user_request,
                outline.title,
                section,
            )
            cleaned = self._clean_plain_text(section_text)
            block = self._plain_text_heading(section) + "\n" + cleaned.strip()
            with report_path.open("a", encoding="utf-8") as handle:
                handle.write(block + "\n\n")
            if chat_settings.debug_enabled and update is not None:
                await self._emit_debug(
                    update,
                    f"section_{index}",
                    {
                        "section": section,
                        "preview": cleaned[:1500],
                        "report_path": str(report_path),
                    },
                )

        final_text = report_path.read_text(encoding="utf-8").strip()
        if chat_settings.debug_enabled and update is not None:
            await self._emit_debug(update, "final_report_path", {"path": str(report_path), "length": len(final_text)})
        return final_text

    async def _build_research_outline(self, prompt_messages: list[dict[str, str]], latest_user_request: str) -> ResearchOutline:
        outline_messages = [
            *prompt_messages,
            {
                "role": "system",
                "content": (
                    "Create a plain-text research report outline. "
                    "Return strict JSON only with keys title and sections. "
                    "The title should be concise and human-readable. "
                    "The sections should be 4 to 7 concrete headings that together answer the user's request thoroughly. "
                    "Prefer headings that stand alone well."
                ),
            },
            {"role": "user", "content": f"Build the report outline for this request: {latest_user_request}"},
        ]
        try:
            raw = await self.ollama_client.chat(outline_messages)
            data = self._extract_json_object(raw)
            outline = ResearchOutline.model_validate(data)
            if not outline.sections:
                raise ValueError("Empty outline sections")
            return outline
        except Exception:
            return ResearchOutline(
                title="Research Report",
                sections=[
                    "What Changed",
                    "Key Trends",
                    "Important Examples",
                    "Practical Takeaways",
                ],
            )

    async def _generate_report_section(
        self,
        prompt_messages: list[dict[str, str]],
        latest_user_request: str,
        title: str,
        section: str,
    ) -> str:
        section_messages = [
            *prompt_messages,
            {
                "role": "system",
                "content": (
                    "Write one section of a research report in plain text. "
                    "Use clean human-readable formatting with short paragraphs and bullet points where they improve clarity. "
                    "Do not use markdown emphasis, bold, italics, decorative formatting, or hype. "
                    "Do not write in an excited, breathless, or promotional tone. "
                    "Avoid all-caps emphasis and avoid lines like 'X is revolutionary'. "
                    "This section must stand on its own and still fit the full report. "
                    "Use only the supplied research context and conversation context. "
                    "Be detailed and concrete rather than terse. "
                    "Prefer this structure when it fits: "
                    "1. a short opening summary paragraph, "
                    "2. 3 to 6 bullet points with the most relevant developments, including dates, names, and specifics when available, "
                    "3. a short 'Why this matters' paragraph, "
                    "4. a short 'Sources used' line naming the strongest sources for that section. "
                    "Only include claims grounded in the provided research packet."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Report title: {title}\n"
                    f"User request: {latest_user_request}\n"
                    f"Write the section titled: {section}"
                ),
            },
        ]
        return await self.ollama_client.chat(section_messages)

    async def _emit_debug(self, update: Update, stage: str, payload: dict) -> None:
        if not update.message:
            return
        message = f"[debug:{stage}]\n```json\n{json.dumps(payload, indent=2)[:3800]}\n```"
        await update.message.reply_text(message)

    @staticmethod
    def _summarize_prompt(messages: list[dict[str, str]]) -> dict:
        return {
            "message_count": len(messages),
            "messages": [
                {
                    "role": message["role"],
                    "preview": message["content"][:600],
                }
                for message in messages[-10:]
            ],
        }

    @staticmethod
    def _clean_plain_text(text: str) -> str:
        cleaned = text.replace("**", "").replace("__", "").replace("```", "")
        cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _plain_text_heading(text: str) -> str:
        return text.strip()

    async def _reply_in_chunks(self, update: Update, text: str) -> None:
        if not update.message:
            return
        for chunk in self._split_text_for_telegram(text):
            await update.message.reply_text(chunk)

    @staticmethod
    def _split_text_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
        stripped = text.strip()
        if len(stripped) <= limit:
            return [stripped] if stripped else [""]

        chunks: list[str] = []
        remaining = stripped
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            split_at = remaining.rfind("\n\n", 0, limit)
            if split_at == -1:
                split_at = remaining.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, limit)
            if split_at == -1 or split_at < limit // 3:
                split_at = limit

            chunk = remaining[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].lstrip()

        return chunks or [stripped]


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
