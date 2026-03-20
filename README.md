# LlamaClaw

LlamaClaw is a Telegram research assistant designed to work with Ollama. It keeps a compact long-term memory file, uses recent chat context plus onboarding profile data, fans research out across multiple worker processes, ranks evidence, and writes section-by-section reports instead of relying on a single one-shot response.

Status: experimental, usable, and still actively being tuned.

## Quick Start

1. Copy [.env.example](C:/Users/evanl/python%20projects/LlamaClaw/.env.example) to `.env`
2. Fill in your Telegram bot token and Brave API key
3. Point `OLLAMA_BASE_URL` at your Ollama server
4. Run:

```powershell
.\scripts\install_local.ps1
.\scripts\run_local.ps1
```

For local Ollama on Windows through Docker Desktop, `OLLAMA_BASE_URL=http://host.docker.internal:11434` is the usual container-safe value. For non-Docker local runs, `http://localhost:11434` also works.

## Features

- Telegram bot with polling by default and webhook support via FastAPI
- Ollama integration with `llama3.2` as the default model
- Brave Search integration for research queries
- Self-review pass that checks draft answers before they are sent
- AI command analysis to decide when to research and what target to use
- Research fan-out pipeline that spawns multiple worker processes, saves temp outputs, and synthesizes them into one final answer
- Evidence extraction and reranking between worker output and report writing
- Research report writer that builds section-by-section in temp files for slower, more reliable long-form outputs
- Section quality loop that scores each section against the original objective and retries until it clears threshold or exhausts retries
- Deeper research context by scraping top search results and crawling a target site for extra evidence
- Runtime timestamp grounding so prompts stay anchored to the current real date and time
- File-backed conversation history and unified memory context
- User onboarding flow that captures goals, preferences, and boundaries
- Per-chat debug tracing and deep-thinking modes
- Async memory refresh every 10 persisted messages by default

## Current Limits

- The default `llama3.2` model works, but stronger Ollama models will usually produce better research synthesis.
- Research quality depends heavily on source quality and the model's ability to stay grounded in the evidence packet.
- The system is optimized for slow, detailed research over fast chatty replies.
- The project is file-backed by design for v1, so large-scale multi-user deployments are not the current target.

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
- `BRAVE_SEARCH_RESULTS`
- `BRAVE_SCRAPE_RESULTS`
- `SITE_CRAWL_MAX_PAGES`
- `SITE_CRAWL_CHARS_PER_PAGE`
- `RESEARCH_PARALLEL_SEARCH_WORKERS`
- `RESEARCH_PARALLEL_CRAWL_WORKERS`
- `SECTION_QUALITY_THRESHOLD`
- `SECTION_QUALITY_MAX_RETRIES`
- `SELF_REVIEW_ENABLED`
- `LOG_LEVEL`
- `RUN_MODE` values: `polling` or `webhook`
- `WEBHOOK_URL`
- `WEBHOOK_SECRET`
- `WEBHOOK_PATH`

## Running locally without Docker

First-time setup:

```powershell
.\scripts\install_local.ps1
```

That script creates `.venv`, installs dependencies, and creates `.env` from `.env.example` if needed.

Run the app:

```bash
llamaclaw
```

On Windows PowerShell, you can also use:

```powershell
.\scripts\run_local.ps1
```

This branch is meant to run directly on your machine, with the local `data/` folder preserving memory over time.

You can also launch it as a Python module:

```bash
python -m app
```

## Telegram Commands

- `/enable`: turn on debug tracing for the current chat
- `/disable`: turn off debug tracing
- `/deepthinking`: enable much broader context and research fan-out for the current chat
- `/normalthinking`: return to the standard research profile
- `/clearcontext` or `/resetcontext`: wipe only the rolling conversation context while keeping onboarding and long-term memory

## Research Flow

For research-style prompts, LlamaClaw currently does this:

1. Decide whether the prompt needs research
2. Build a research plan with intent-aware search queries
3. Spawn parallel worker processes that search and optionally crawl sites
4. Save worker outputs to temp files
5. Extract and rank evidence from those worker results
6. Build a report outline
7. Draft each report section separately
8. Score each section against the original objective and retry weak sections
9. Append the final report to a temp file and send it back to Telegram in chunks

## Data And Privacy

LlamaClaw stores local runtime data under `data/`, including chat transcripts, onboarding profiles, memory state, and system prompt files. That directory is intentionally ignored by git so the repo can be public while your learned assistant state stays local.

Tracked in git:

- app code
- tests
- scripts
- `.env.example`
- data folder scaffold

Not tracked in git:

- `.env`
- live conversation history
- learned memory files
- onboarding/profile data

## Publishing Notes

If you publish this repo:

- keep [.env.example](C:/Users/evanl/python%20projects/LlamaClaw/.env.example) as the setup template
- do not commit `.env`
- do not commit generated files under `data/conversations`, `data/memory`, `data/profiles`, or `data/state`
- consider rotating any tokens that were ever pasted into chat or local logs during testing

## Optional Docker

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
