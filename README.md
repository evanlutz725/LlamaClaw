# LlamaClaw

LlamaClaw is a lightweight Telegram chatbot and research assistant designed to work with Ollama. It keeps a compact long-term memory file, uses the last 20 chat turns for short-term context, and can call Brave Search for research-style replies.

## Features

- Telegram bot with polling by default and webhook support via FastAPI
- Ollama integration with `llama3.2` as the default model
- Brave Search integration for research queries
- Self-review pass that checks draft answers before they are sent
- AI command analysis to decide when to research and what target to use
- Deeper research context by scraping top search results and crawling a target site for extra evidence
- File-backed conversation history and unified memory context
- User onboarding flow that captures goals, preferences, and boundaries
- Async memory refresh every 10 persisted messages by default

## Environment

Set these environment variables before starting the app:

- `TELEGRAM_BOT_TOKEN`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL` default: `llama3.2`
- `BRAVE_API_KEY`
- `SYSTEM_PROMPT_PATH`
- `DATA_DIR`
- `CHAT_WINDOW_SIZE`
- `MEMORY_REFRESH_EVERY_MESSAGES`
- `MEMORY_MIN_WEIGHT`
- `MEMORY_RETENTION_DAYS`
- `RUN_MODE` values: `polling` or `webhook`
- `WEBHOOK_URL`
- `WEBHOOK_SECRET`
- `WEBHOOK_PATH`

## Running locally

```bash
pip install -e .[dev]
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t llamaclaw .
docker run --rm -p 8000:8000 --env-file .env llamaclaw
```

## Docker Compose With Persistent Memory

Use Docker Compose to keep learned memory on your machine across container rebuilds and restarts:

```bash
docker compose up --build
```

The local `data/` folder is mounted into `/app/data`, so profiles, conversations, and long-term memory persist outside the container.
