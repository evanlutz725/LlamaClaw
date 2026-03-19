from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from telegram import Update

from app.bot import LlamaClawBot, ensure_default_system_prompt
from app.clients import BraveSearchClient, OllamaClient
from app.config import get_settings
from app.repositories import (
    ChatSettingsRepository,
    ConversationRepository,
    MemoryRepository,
    OnboardingRepository,
    RefreshStateRepository,
    UserProfileRepository,
)
from app.services import ContextAssembler, MemoryRefreshWorker, MemoryScorer
from app.storage import JsonFileStore


def create_bot() -> LlamaClawBot:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    store = JsonFileStore()
    system_prompt = ensure_default_system_prompt(settings, store)

    conversations = ConversationRepository(settings.data_dir, store)
    memory_repo = MemoryRepository(settings.data_dir, store)
    profile_repo = UserProfileRepository(settings.data_dir, store)
    onboarding_repo = OnboardingRepository(settings.data_dir, store)
    chat_settings_repo = ChatSettingsRepository(settings.data_dir, store)
    refresh_repo = RefreshStateRepository(settings.data_dir, store)
    scorer = MemoryScorer(settings.memory_retention_days, settings.memory_min_weight)
    assembler = ContextAssembler(system_prompt, settings.chat_window_size)
    refresh_worker = MemoryRefreshWorker(
        conversations=conversations,
        memory_repo=memory_repo,
        refresh_repo=refresh_repo,
        scorer=scorer,
        refresh_every_messages=settings.memory_refresh_every_messages,
    )
    return LlamaClawBot(
        settings=settings,
        conversation_repo=conversations,
        memory_repo=memory_repo,
        profile_repo=profile_repo,
        onboarding_repo=onboarding_repo,
        chat_settings_repo=chat_settings_repo,
        context_assembler=assembler,
        ollama_client=OllamaClient(settings.ollama_base_url, settings.ollama_model),
        brave_client=BraveSearchClient(settings.brave_api_key),
        refresh_worker=refresh_worker,
    )


bot = create_bot()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if bot.settings.run_mode == "polling":
        await bot.application.initialize()
        await bot.application.start()
        await bot.application.updater.start_polling()
    elif bot.settings.run_mode == "webhook":
        await bot.application.initialize()
        await bot.application.start()
        if bot.settings.webhook_url:
            await bot.application.bot.set_webhook(
                url=bot.settings.webhook_url.rstrip("/") + bot.settings.webhook_path,
                secret_token=bot.settings.webhook_secret or None,
            )

    yield

    if bot.settings.run_mode == "polling":
        await bot.application.updater.stop()
    elif bot.settings.run_mode == "webhook" and bot.settings.webhook_url:
        await bot.application.bot.delete_webhook()
    await bot.application.stop()
    await bot.application.shutdown()


app = FastAPI(title="LlamaClaw", lifespan=lifespan)


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    if bot.settings.run_mode != "webhook":
        raise HTTPException(status_code=400, detail="Webhook mode is disabled.")
    if bot.settings.webhook_secret and x_telegram_bot_api_secret_token != bot.settings.webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid webhook secret.")

    payload = await request.json()
    update = Update.de_json(payload, bot.application.bot)
    await bot.application.process_update(update)
    return {"status": "accepted"}


app.add_api_route(bot.settings.webhook_path, telegram_webhook, methods=["POST"])
