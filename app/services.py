from __future__ import annotations

import asyncio
import math
import re
import uuid
from collections import defaultdict
from datetime import datetime

from app.models import ChatMessage, MemoryCandidate, MemoryItem, MemoryStore, UserProfile, utc_now
from app.repositories import ConversationRepository, MemoryRepository, RefreshStateRepository


class ContextAssembler:
    def __init__(self, system_prompt: str, chat_window_size: int) -> None:
        self._system_prompt = system_prompt
        self._chat_window_size = chat_window_size

    def build_messages(
        self,
        memory: MemoryStore,
        conversation: list[ChatMessage],
        profile: UserProfile | None = None,
        research_context: str | None = None,
    ) -> list[dict[str, str]]:
        system_content = self._system_prompt
        if profile:
            profile_summary = self._build_profile_summary(profile)
            if profile_summary:
                system_content += "\n\nKnown user profile:\n" + profile_summary

        system_content += (
            "\n\nRuntime capabilities:\n"
            "- You can access the internet through Brave Search when the app supplies search results.\n"
            "- You can inspect multiple pages from the same website when the app supplies crawled site pages.\n"
            "- If Brave results or page excerpts are present, say you used web research and rely on those results.\n"
            "- Do not claim you cannot browse the web when research context is available.\n"
            "- You have persistent local memory through saved conversation files, onboarding profile data, and the unified context file.\n"
            "- Be honest about the difference between direct web results you were given and anything you infer from them."
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        active_items = [item for item in memory.items if item.status == "active" and item.combined_weight > 0]
        if active_items:
            memory_lines = ["Unified previous context:"]
            for item in sorted(active_items, key=lambda current: current.combined_weight, reverse=True):
                memory_lines.append(
                    f"- {item.title} (weight={item.combined_weight:.2f}, first_seen={item.first_seen_at.date()}, last_seen={item.last_seen_at.date()})"
                )
                for bullet in item.bullets:
                    memory_lines.append(f"  - {bullet}")
            messages.append({"role": "system", "content": "\n".join(memory_lines)})

        if research_context:
            messages.append({"role": "system", "content": research_context})

        recent_messages = conversation[-self._chat_window_size :]
        for message in recent_messages:
            messages.append({"role": message.role, "content": message.text})
        return messages

    @staticmethod
    def _build_profile_summary(profile: UserProfile) -> str:
        if not profile.onboarding_complete:
            return ""
        lines: list[str] = []
        if profile.preferred_name:
            lines.append(f"- Preferred name: {profile.preferred_name}")
        if profile.primary_goals:
            lines.append(f"- Primary goals: {', '.join(profile.primary_goals)}")
        if profile.active_projects:
            lines.append(f"- Active projects: {', '.join(profile.active_projects)}")
        if profile.family_context:
            lines.append(f"- Family context: {', '.join(profile.family_context)}")
        if profile.preferences:
            lines.append(f"- Preferences: {', '.join(profile.preferences)}")
        if profile.boundaries:
            lines.append(f"- Boundaries: {', '.join(profile.boundaries)}")
        if profile.research_preferences:
            lines.append(f"- Research preferences: {', '.join(profile.research_preferences)}")
        return "\n".join(lines)


class MemoryScorer:
    def __init__(self, retention_days: int, min_weight: float) -> None:
        self._retention_days = retention_days
        self._min_weight = min_weight

    def score_age(self, seen_at: datetime, now: datetime | None = None) -> float:
        current = now or utc_now()
        age_days = max((current - seen_at).days, 0)
        if age_days <= 7:
            return 1.0
        if age_days >= self._retention_days:
            return 0.0
        normalized = age_days / self._retention_days
        return max(0.0, round(math.exp(-3 * normalized), 4))

    def score_importance(self, text: str) -> float:
        lower = text.lower()
        score = 0.25
        durable_terms = [
            "project",
            "goal",
            "plan",
            "family",
            "kids",
            "child",
            "wife",
            "husband",
            "relationship",
            "prefer",
            "always",
            "never",
            "important",
            "college",
            "work",
            "business",
            "brand",
        ]
        transient_terms = [
            "today",
            "yesterday",
            "lol",
            "maybe",
            "random",
            "temporary",
            "one-time",
            "bored",
        ]
        if any(token in lower for token in durable_terms):
            score += 0.45
        if re.search(r"\b(i am|i'm|my|we need|i need|remember)\b", lower):
            score += 0.15
        if any(token in lower for token in transient_terms):
            score -= 0.15
        if len(text.split()) >= 8:
            score += 0.1
        if any(token in lower for token in transient_terms) and not any(token in lower for token in durable_terms):
            score -= 0.3
        if len(text.split()) <= 4 and not any(token in lower for token in durable_terms):
            score -= 0.15
        return max(0.0, min(1.0, round(score, 4)))

    def combined_weight(self, importance_score: float, age_score: float) -> float:
        if importance_score <= 0:
            return 0.0
        combined = (importance_score * 0.7) + (age_score * 0.3)
        if combined < self._min_weight:
            return 0.0
        return round(min(combined, 1.0), 4)


class MemoryRefreshWorker:
    def __init__(
        self,
        conversations: ConversationRepository,
        memory_repo: MemoryRepository,
        refresh_repo: RefreshStateRepository,
        scorer: MemoryScorer,
        refresh_every_messages: int,
    ) -> None:
        self._conversations = conversations
        self._memory_repo = memory_repo
        self._refresh_repo = refresh_repo
        self._scorer = scorer
        self._refresh_every_messages = refresh_every_messages
        self._refresh_lock = asyncio.Lock()
        self._pending_task: asyncio.Task[None] | None = None

    def record_message_and_maybe_refresh(self, chat_id: str) -> None:
        state = self._refresh_repo.load()
        state.messages_since_refresh += 1
        self._refresh_repo.save(state)
        if state.messages_since_refresh >= self._refresh_every_messages:
            if self._pending_task and not self._pending_task.done():
                return
            self._pending_task = asyncio.create_task(self.refresh())

    async def refresh(self) -> None:
        async with self._refresh_lock:
            state = self._refresh_repo.load()
            memory = self._memory_repo.load()
            now = utc_now()

            conversations_dir = self._conversations.data_dir / "conversations"
            candidates: list[MemoryCandidate] = []
            if conversations_dir.exists():
                for path in sorted(conversations_dir.glob("*.json")):
                    chat_id = path.stem
                    record = self._conversations.load(chat_id)
                    start_index = state.last_processed_offsets.get(chat_id, 0)
                    new_messages = record.messages[start_index:]
                    candidates.extend(self._extract_candidates(chat_id, new_messages))
                    state.last_processed_offsets[chat_id] = len(record.messages)

            merged = self._merge(memory, candidates, now)
            self._memory_repo.save(merged)
            state.messages_since_refresh = 0
            state.last_refresh_at = now
            self._refresh_repo.save(state)

    def _extract_candidates(self, chat_id: str, messages: list[ChatMessage]) -> list[MemoryCandidate]:
        grouped: dict[str, list[ChatMessage]] = defaultdict(list)
        for message in messages:
            if message.role != "user":
                continue
            title = self._derive_title(message.text)
            grouped[title].append(message)

        candidates: list[MemoryCandidate] = []
        for title, grouped_messages in grouped.items():
            bullets: list[str] = []
            combined_text = " ".join(message.text for message in grouped_messages)
            for message in grouped_messages[:4]:
                bullets.extend(self._extract_bullets(message.text))
            unique_bullets = list(dict.fromkeys(bullets))[:6]
            if not unique_bullets:
                unique_bullets = [grouped_messages[-1].text.strip()]
            candidates.append(
                MemoryCandidate(
                    title=title,
                    bullets=unique_bullets,
                    source_chat_id=chat_id,
                    first_seen_at=grouped_messages[0].created_at,
                    last_seen_at=grouped_messages[-1].created_at,
                    importance_score=self._scorer.score_importance(combined_text),
                )
            )
        return candidates

    def _merge(self, memory: MemoryStore, candidates: list[MemoryCandidate], now: datetime) -> MemoryStore:
        items = {item.title.lower(): item for item in memory.items}

        for candidate in candidates:
            key = candidate.title.lower()
            if key in items:
                item = items[key]
                item.last_seen_at = max(item.last_seen_at, candidate.last_seen_at)
                item.first_seen_at = min(item.first_seen_at, candidate.first_seen_at)
                item.source_chat_id = candidate.source_chat_id
                item.importance_score = max(item.importance_score, candidate.importance_score)
                item.bullets = list(dict.fromkeys([*item.bullets, *candidate.bullets]))[:8]
            else:
                items[key] = MemoryItem(
                    id=str(uuid.uuid4()),
                    title=candidate.title,
                    bullets=candidate.bullets,
                    source_chat_id=candidate.source_chat_id,
                    first_seen_at=candidate.first_seen_at,
                    last_seen_at=candidate.last_seen_at,
                    last_referenced_at=None,
                    importance_score=candidate.importance_score,
                    age_score=0.0,
                    combined_weight=0.0,
                    status="active",
                )

        updated_items: list[MemoryItem] = []
        for item in items.values():
            item.age_score = self._scorer.score_age(item.last_seen_at, now=now)
            item.combined_weight = self._scorer.combined_weight(item.importance_score, item.age_score)
            item.status = "discarded" if item.combined_weight <= 0 else "active"
            updated_items.append(item)
        updated_items.sort(key=lambda current: (current.status != "active", -current.combined_weight, current.title.lower()))
        return MemoryStore(items=updated_items)

    @staticmethod
    def _derive_title(text: str) -> str:
        sanitized = re.sub(r"[\r\n]+", " ", text).strip(" -")
        words = sanitized.split()
        return " ".join(words[:6]).strip().title() or "Untitled Memory"

    @staticmethod
    def _extract_bullets(text: str) -> list[str]:
        lines = [segment.strip(" -") for segment in re.split(r"[\n.;]", text) if segment.strip()]
        return [line for line in lines if len(line.split()) >= 3][:5]


class OnboardingService:
    QUESTIONS: list[tuple[str, str]] = [
        ("preferred_name", "What should I call you?"),
        ("primary_goals", "What are the main goals you want my help with right now?"),
        ("active_projects", "What projects, businesses, or responsibilities are active in your life?"),
        ("family_context", "What family or personal context should I remember?"),
        ("preferences", "How do you like me to respond: concise, detailed, tactical, creative, direct, or something else?"),
        ("boundaries", "What should I avoid or be careful about when helping you?"),
        ("research_preferences", "When I do research, do you want quick summaries, source links, deep dives, or a mix?"),
    ]

    @classmethod
    def first_prompt(cls) -> str:
        return (
            "I can learn your preferences and priorities before we get into normal chat. "
            "Let's do a quick onboarding.\n\n"
            f"1/{len(cls.QUESTIONS)}: {cls.QUESTIONS[0][1]}"
        )

    @classmethod
    def next_question(cls, step_index: int) -> str | None:
        if step_index >= len(cls.QUESTIONS):
            return None
        _, question = cls.QUESTIONS[step_index]
        return f"{step_index + 1}/{len(cls.QUESTIONS)}: {question}"

    @classmethod
    def record_answer(cls, state, profile: UserProfile, answer: str) -> tuple[UserProfile, object, str | None]:
        key, _ = cls.QUESTIONS[state.current_step]
        state.answers[key] = answer.strip()
        state.current_step += 1
        state.updated_at = utc_now()

        if state.current_step >= len(cls.QUESTIONS):
            cls._apply_answers(profile, state.answers)
            profile.onboarding_complete = True
            profile.updated_at = utc_now()
            state.completed = True
            state.started = True
            return profile, state, None
        return profile, state, cls.next_question(state.current_step)

    @staticmethod
    def _split_list(raw: str) -> list[str]:
        return [part.strip() for part in re.split(r"[\n,;]", raw) if part.strip()]

    @classmethod
    def _apply_answers(cls, profile: UserProfile, answers: dict[str, str]) -> None:
        profile.preferred_name = answers.get("preferred_name")
        profile.primary_goals = cls._split_list(answers.get("primary_goals", ""))
        profile.active_projects = cls._split_list(answers.get("active_projects", ""))
        profile.family_context = cls._split_list(answers.get("family_context", ""))
        profile.preferences = cls._split_list(answers.get("preferences", ""))
        profile.boundaries = cls._split_list(answers.get("boundaries", ""))
        profile.research_preferences = cls._split_list(answers.get("research_preferences", ""))
