from __future__ import annotations

from pathlib import Path

from app.models import ConversationRecord, MemoryStore, OnboardingState, RefreshState, UserProfile
from app.storage import JsonFileStore


class ConversationRepository:
    def __init__(self, data_dir: Path, store: JsonFileStore) -> None:
        self.data_dir = data_dir
        self._store = store

    def path_for_chat(self, chat_id: str) -> Path:
        return self.data_dir / "conversations" / f"{chat_id}.json"

    def load(self, chat_id: str) -> ConversationRecord:
        return self._store.read_model(
            self.path_for_chat(chat_id),
            ConversationRecord,
            ConversationRecord(chat_id=chat_id),
        )

    def save(self, record: ConversationRecord) -> None:
        self._store.write_model(self.path_for_chat(record.chat_id), record)


class MemoryRepository:
    def __init__(self, data_dir: Path, store: JsonFileStore) -> None:
        self._path = data_dir / "memory" / "unified_context.json"
        self._store = store

    def load(self) -> MemoryStore:
        return self._store.read_model(self._path, MemoryStore, MemoryStore())

    def save(self, memory: MemoryStore) -> None:
        self._store.write_model(self._path, memory)


class RefreshStateRepository:
    def __init__(self, data_dir: Path, store: JsonFileStore) -> None:
        self._path = data_dir / "state" / "refresh_state.json"
        self._store = store

    def load(self) -> RefreshState:
        return self._store.read_model(self._path, RefreshState, RefreshState())

    def save(self, state: RefreshState) -> None:
        self._store.write_model(self._path, state)


class UserProfileRepository:
    def __init__(self, data_dir: Path, store: JsonFileStore) -> None:
        self.data_dir = data_dir
        self._store = store

    def _path(self, chat_id: str) -> Path:
        return self.data_dir / "profiles" / f"{chat_id}.json"

    def load(self, chat_id: str) -> UserProfile:
        return self._store.read_model(self._path(chat_id), UserProfile, UserProfile(chat_id=chat_id))

    def save(self, profile: UserProfile) -> None:
        self._store.write_model(self._path(profile.chat_id), profile)


class OnboardingRepository:
    def __init__(self, data_dir: Path, store: JsonFileStore) -> None:
        self.data_dir = data_dir
        self._store = store

    def _path(self, chat_id: str) -> Path:
        return self.data_dir / "state" / "onboarding" / f"{chat_id}.json"

    def load(self, chat_id: str) -> OnboardingState:
        return self._store.read_model(self._path(chat_id), OnboardingState, OnboardingState(chat_id=chat_id))

    def save(self, state: OnboardingState) -> None:
        self._store.write_model(self._path(state.chat_id), state)
