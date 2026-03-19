from datetime import timedelta
from types import SimpleNamespace

from app.bot import LlamaClawBot
from app.models import ChatMessage, CommandDecision, MemoryItem, MemoryStore, UserProfile, utc_now
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

    assert messages[0]["content"].startswith("system prompt")
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


def test_context_includes_runtime_capabilities() -> None:
    assembler = ContextAssembler("system prompt", chat_window_size=20)

    messages = assembler.build_messages(MemoryStore(), [])

    assert "Runtime capabilities" in messages[0]["content"]
    assert "You can access the internet through Brave Search" in messages[0]["content"]
    assert "multiple pages from the same website" in messages[0]["content"]


class FakeOllamaClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.responses[len(self.calls) - 1]


async def test_self_review_runs_second_pass_when_enabled() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.settings = SimpleNamespace(self_review_enabled=True)
    bot.ollama_client = FakeOllamaClient(["draft answer", "reviewed answer"])

    result = await bot._generate_response([{"role": "system", "content": "prompt"}], [])

    assert result == "reviewed answer"
    assert len(bot.ollama_client.calls) == 2
    assert any(message["content"] == "draft answer" for message in bot.ollama_client.calls[1])


async def test_self_review_can_be_disabled() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.settings = SimpleNamespace(self_review_enabled=False)
    bot.ollama_client = FakeOllamaClient(["draft answer"])

    result = await bot._generate_response([{"role": "system", "content": "prompt"}], [])

    assert result == "draft answer"
    assert len(bot.ollama_client.calls) == 1


async def test_command_decision_uses_model_output() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(['{"cmd":"research","search":"bowtiedcyber","url":null,"reason":"user asked for investigation"}'])

    decision = await bot._decide_command("Do research on bowtiedcyber and see what you find")

    assert decision == CommandDecision(cmd="research", search="bowtiedcyber", url=None, reason="user asked for investigation")


async def test_command_decision_falls_back_to_url_research() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(["not json"])

    decision = await bot._decide_command("Tell me about https://example.com")

    assert decision.cmd == "research"
    assert decision.url == "https://example.com"
