from __future__ import annotations

import logging
import re
import uuid

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.clients import BraveSearchClient, OllamaClient
from app.config import Settings
from app.models import ChatMessage
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

        research_context = None
        search_results = []
        if self._is_research_query(user_text):
            search_results = await self.brave_client.search(self._research_query(user_text))
            formatted_results = self.brave_client.format_results(search_results)
            research_context = "Brave Search results:\n" + formatted_results

        memory = self.memory_repo.load()
        prompt_messages = self.context_assembler.build_messages(
            memory,
            conversation.messages,
            profile=profile,
            research_context=research_context,
        )
        response_text = await self.ollama_client.chat(prompt_messages)
        response_text = self._append_sources(response_text, search_results)

        conversation.messages.append(ChatMessage(id=str(uuid.uuid4()), role="assistant", text=response_text))
        self.conversation_repo.save(conversation)
        self.refresh_worker.record_message_and_maybe_refresh(chat_id)
        await update.message.reply_text(response_text)

    def _is_research_query(self, text: str) -> bool:
        lower = text.lower().strip()
        return lower.startswith("research:") or lower.startswith("/research") or bool(re.search(r"\b(search|research|look up|find online)\b", lower))

    @staticmethod
    def _research_query(text: str) -> str:
        if ":" in text:
            return text.split(":", 1)[1].strip()
        return text.strip()

    @staticmethod
    def _append_sources(response_text: str, search_results: list) -> str:
        if not search_results:
            return response_text
        source_lines = ["", "Sources:"]
        for result in search_results[:5]:
            source_lines.append(f"- {result.title}: {result.url}")
        return response_text.strip() + "\n" + "\n".join(source_lines)


def ensure_default_system_prompt(settings: Settings, store: JsonFileStore) -> str:
    return store.read_text(
        settings.system_prompt_path,
        default=(
            "You are LlamaClaw, a concise Telegram chatbot and research assistant. "
            "You have access to persisted local conversation files and a unified long-term memory file. "
            "You can use Brave Search when research results are supplied. "
            "Use the supplied memory, onboarding profile, and recent conversation context. "
            "When research results are present, ground your response in them and cite sources plainly."
        ),
    )
