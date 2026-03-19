from __future__ import annotations

import json
import logging
import re
import uuid
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.clients import BraveSearchClient, OllamaClient
from app.config import Settings
from app.models import ChatMessage, CommandDecision
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
        await update.message.reply_text("LlamaClaw is online. Send a message or start with 'research:' for web-backed replies.")

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

        command = await self._decide_command(user_text)
        research_context = None
        search_results = []
        crawled_pages = []
        if command.cmd == "research":
            research_query = command.search or self._research_query(user_text)
            search_results = await self.brave_client.search(
                research_query,
                count=self.settings.brave_search_results,
            )
            search_results = await self.brave_client.enrich_with_page_content(
                search_results,
                max_fetch=self.settings.brave_scrape_results,
                max_chars=self.settings.site_crawl_chars_per_page,
            )
            crawl_target = command.url or self._select_crawl_target(search_results)
            if crawl_target:
                crawled_pages = await self.brave_client.crawl_site(
                    crawl_target,
                    max_pages=self.settings.site_crawl_max_pages,
                    max_chars=self.settings.site_crawl_chars_per_page,
                )
            formatted_results = self.brave_client.format_results(search_results)
            formatted_pages = self.brave_client.format_site_pages(crawled_pages)
            research_context = (
                "Internet access is available for this reply through Brave Search.\n"
                "Use the following web results and page excerpts as your browsing context.\n"
                "When site crawl pages are present, synthesize across the whole crawled site rather than treating it as a single page.\n"
                "Be explicit that you are using web research when relevant.\n\n"
                f"Command decision: {command.model_dump_json(exclude_none=True)}\n\n"
                "Brave Search results:\n"
                + formatted_results
                + "\n\nCrawled site pages:\n"
                + formatted_pages
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

    async def _decide_command(self, text: str) -> CommandDecision:
        direct_url = self._extract_url(text)
        planner_messages = [
            {
                "role": "system",
                "content": (
                    "You classify the user's next requested action for an assistant. "
                    "Return strict JSON only with keys cmd, search, url, reason. "
                    "cmd must be either chat or research. "
                    "Use research when the user is asking for facts, investigation, online discovery, external validation, or to look into a person, brand, site, or company. "
                    "Set search to the best concise search query. "
                    "Set url if the user gave a direct URL or clearly wants a specific website crawled. "
                    "Do not include markdown fences or extra commentary."
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

    @staticmethod
    def _append_sources(response_text: str, search_results: list) -> str:
        if not search_results:
            return response_text
        source_lines = ["", "Sources:"]
        for result in search_results[:5]:
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
            "When research results are present, ground your response in them, acknowledge that you used web research, and cite sources plainly."
        ),
    )
