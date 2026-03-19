from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    id: str
    role: Literal["system", "user", "assistant", "tool"]
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class ConversationRecord(BaseModel):
    chat_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class MemoryItem(BaseModel):
    id: str
    title: str
    bullets: list[str]
    source_chat_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_referenced_at: datetime | None = None
    importance_score: float
    age_score: float
    combined_weight: float
    status: Literal["active", "discarded"] = "active"


class MemoryStore(BaseModel):
    items: list[MemoryItem] = Field(default_factory=list)


class RefreshState(BaseModel):
    messages_since_refresh: int = 0
    last_refresh_at: datetime | None = None
    last_processed_offsets: dict[str, int] = Field(default_factory=dict)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    page_excerpt: str | None = None


class SitePage(BaseModel):
    url: str
    title: str | None = None
    excerpt: str


class MemoryCandidate(BaseModel):
    title: str
    bullets: list[str]
    source_chat_id: str
    first_seen_at: datetime
    last_seen_at: datetime
    importance_score: float


class UserProfile(BaseModel):
    chat_id: str
    display_name: str | None = None
    preferred_name: str | None = None
    primary_goals: list[str] = Field(default_factory=list)
    active_projects: list[str] = Field(default_factory=list)
    family_context: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    research_preferences: list[str] = Field(default_factory=list)
    onboarding_complete: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class OnboardingState(BaseModel):
    chat_id: str
    started: bool = False
    current_step: int = 0
    answers: dict[str, str] = Field(default_factory=dict)
    completed: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class CommandDecision(BaseModel):
    cmd: Literal["chat", "research"] = "chat"
    search: str | None = None
    url: str | None = None
    reason: str | None = None


class ResearchPlan(BaseModel):
    intent_type: Literal["general", "news", "company", "leadgen", "technical"] = "general"
    search_queries: list[str] = Field(default_factory=list)
    crawl_urls: list[str] = Field(default_factory=list)
    goal: str | None = None


class ResearchOutline(BaseModel):
    title: str
    sections: list[str] = Field(default_factory=list)


class ResearchEvidence(BaseModel):
    claim: str
    source: str
    date: str | None = None
    relevance_score: float = 0.0
    notes: str | None = None


class WorkerResult(BaseModel):
    worker_type: Literal["search", "crawl"]
    target: str
    output_path: str
    summary: str


class ChatSettings(BaseModel):
    chat_id: str
    debug_enabled: bool = False
    deep_thinking_enabled: bool = False
    updated_at: datetime = Field(default_factory=utc_now)
