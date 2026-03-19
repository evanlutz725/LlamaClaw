import asyncio
from pathlib import Path

from app.models import ChatMessage
from app.repositories import ConversationRepository, MemoryRepository, RefreshStateRepository
from app.services import MemoryRefreshWorker, MemoryScorer
from app.storage import JsonFileStore


async def test_refresh_merges_topics_and_discards_low_weight(tmp_path: Path) -> None:
    store = JsonFileStore()
    conversations = ConversationRepository(tmp_path, store)
    memory_repo = MemoryRepository(tmp_path, store)
    refresh_repo = RefreshStateRepository(tmp_path, store)
    scorer = MemoryScorer(retention_days=30, min_weight=0.2)
    worker = MemoryRefreshWorker(
        conversations=conversations,
        memory_repo=memory_repo,
        refresh_repo=refresh_repo,
        scorer=scorer,
        refresh_every_messages=2,
    )

    record = conversations.load("123")
    record.messages.extend(
        [
            ChatMessage(id="1", role="user", text="Spend more time with kids. Johnny is going to college soon."),
            ChatMessage(id="2", role="assistant", text="Let's make a plan."),
            ChatMessage(id="3", role="user", text="Spend more time with kids and protect family evenings."),
            ChatMessage(id="4", role="user", text="lol random temporary thing"),
        ]
    )
    conversations.save(record)

    await worker.refresh()
    memory = memory_repo.load()

    active_titles = [item.title for item in memory.items if item.status == "active"]
    discarded_titles = [item.title for item in memory.items if item.status == "discarded"]

    assert any("Spend More Time With Kids" in title for title in active_titles)
    assert any("Lol Random Temporary Thing" in title for title in discarded_titles)


async def test_refresh_trigger_creates_background_task(tmp_path: Path) -> None:
    store = JsonFileStore()
    conversations = ConversationRepository(tmp_path, store)
    memory_repo = MemoryRepository(tmp_path, store)
    refresh_repo = RefreshStateRepository(tmp_path, store)
    scorer = MemoryScorer(retention_days=30, min_weight=0.2)
    worker = MemoryRefreshWorker(
        conversations=conversations,
        memory_repo=memory_repo,
        refresh_repo=refresh_repo,
        scorer=scorer,
        refresh_every_messages=2,
    )

    worker.record_message_and_maybe_refresh("123")
    worker.record_message_and_maybe_refresh("123")
    await asyncio.sleep(0)

    state = refresh_repo.load()
    assert state.messages_since_refresh in {0, 2}


def test_reset_chat_context_resets_offset_only(tmp_path: Path) -> None:
    store = JsonFileStore()
    conversations = ConversationRepository(tmp_path, store)
    memory_repo = MemoryRepository(tmp_path, store)
    refresh_repo = RefreshStateRepository(tmp_path, store)
    scorer = MemoryScorer(retention_days=30, min_weight=0.2)
    worker = MemoryRefreshWorker(
        conversations=conversations,
        memory_repo=memory_repo,
        refresh_repo=refresh_repo,
        scorer=scorer,
        refresh_every_messages=2,
    )

    state = refresh_repo.load()
    state.last_processed_offsets["123"] = 9
    refresh_repo.save(state)

    worker.reset_chat_context("123")
    updated_state = refresh_repo.load()

    assert updated_state.last_processed_offsets["123"] == 0
