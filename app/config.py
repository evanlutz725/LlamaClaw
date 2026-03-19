from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    system_prompt_path: Path = Field(default=Path("data/system/system_prompt.txt"), alias="SYSTEM_PROMPT_PATH")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    chat_window_size: int = Field(default=20, alias="CHAT_WINDOW_SIZE")
    memory_refresh_every_messages: int = Field(default=10, alias="MEMORY_REFRESH_EVERY_MESSAGES")
    memory_min_weight: float = Field(default=0.15, alias="MEMORY_MIN_WEIGHT")
    memory_retention_days: int = Field(default=180, alias="MEMORY_RETENTION_DAYS")
    brave_search_results: int = Field(default=8, alias="BRAVE_SEARCH_RESULTS")
    brave_scrape_results: int = Field(default=3, alias="BRAVE_SCRAPE_RESULTS")
    site_crawl_max_pages: int = Field(default=8, alias="SITE_CRAWL_MAX_PAGES")
    site_crawl_chars_per_page: int = Field(default=1800, alias="SITE_CRAWL_CHARS_PER_PAGE")
    self_review_enabled: bool = Field(default=True, alias="SELF_REVIEW_ENABLED")
    run_mode: str = Field(default="polling", alias="RUN_MODE")
    webhook_url: str = Field(default="", alias="WEBHOOK_URL")
    webhook_secret: str = Field(default="", alias="WEBHOOK_SECRET")
    webhook_path: str = Field(default="/telegram/webhook", alias="WEBHOOK_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
