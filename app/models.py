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
