from datetime import timedelta

from app.models import ChatMessage, MemoryItem, MemoryStore, UserProfile, utc_now
from app.services import ContextAssembler, MemoryScorer


def test_context_order_and_window_limit() -> None:
    assembler = ContextAssembler("system prompt", chat_window_size=20)
    now = utc_now()
    memory = MemoryStore(
        items=[
            MemoryItem(
                id="1",
                title="Important project",
                bullets=["Ship the funnel", "Keep the rock creative"],
                source_chat_id="1",
                first_seen_at=now,
                last_seen_at=now,
                last_referenced_at=None,
                importance_score=0.9,
                age_score=1.0,
                combined_weight=0.93,
                status="active",
            )
        ]
    )
    conversation = [ChatMessage(id=str(index), role="user" if index % 2 == 0 else "assistant", text=f"message {index}") for index in range(30)]

    messages = assembler.build_messages(memory, conversation)

    assert messages[0]["content"] == "system prompt"
    assert messages[1]["content"].startswith("Unified previous context:")
    assert len(messages) == 22
    assert messages[2]["content"] == "message 10"
    assert messages[-1]["content"] == "message 29"


def test_age_and_combined_scoring() -> None:
    scorer = MemoryScorer(retention_days=180, min_weight=0.15)
    recent_score = scorer.score_age(utc_now() - timedelta(days=2))
    old_score = scorer.score_age(utc_now() - timedelta(days=220))

    assert recent_score == 1.0
    assert old_score == 0.0
    assert scorer.combined_weight(importance_score=0.7, age_score=recent_score) > 0
    assert scorer.combined_weight(importance_score=0.1, age_score=0.0) == 0.0


def test_context_includes_completed_profile_summary() -> None:
    assembler = ContextAssembler("system prompt", chat_window_size=20)
    profile = UserProfile(
        chat_id="1",
        preferred_name="Evan",
        primary_goals=["Build LlamaClaw"],
        preferences=["direct", "concise"],
        onboarding_complete=True,
    )

    messages = assembler.build_messages(MemoryStore(), [], profile=profile)

    assert "Known user profile" in messages[0]["content"]
    assert "Preferred name: Evan" in messages[0]["content"]
