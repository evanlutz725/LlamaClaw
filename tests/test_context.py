from datetime import timedelta
from types import SimpleNamespace

from app.bot import LlamaClawBot
from app.models import ChatMessage, CommandDecision, MemoryItem, MemoryStore, ResearchEvidence, ResearchOutline, ResearchPlan, UserProfile, utc_now
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


def test_context_includes_current_timestamp() -> None:
    assembler = ContextAssembler("system prompt", chat_window_size=20)

    messages = assembler.build_messages(MemoryStore(), [])

    assert "Current timestamp" in messages[0]["content"]
    assert "Local date:" in messages[0]["content"]


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
    chat_settings = SimpleNamespace(debug_enabled=False)

    result = await bot._generate_response([{"role": "system", "content": "prompt"}], [], chat_settings)

    assert result == "reviewed answer"
    assert len(bot.ollama_client.calls) == 2
    assert any(message["content"] == "draft answer" for message in bot.ollama_client.calls[1])


async def test_self_review_can_be_disabled() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.settings = SimpleNamespace(self_review_enabled=False)
    bot.ollama_client = FakeOllamaClient(["draft answer"])
    chat_settings = SimpleNamespace(debug_enabled=False)

    result = await bot._generate_response([{"role": "system", "content": "prompt"}], [], chat_settings)

    assert result == "draft answer"
    assert len(bot.ollama_client.calls) == 1


async def test_self_review_targets_latest_user_request() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.settings = SimpleNamespace(self_review_enabled=True)
    bot.ollama_client = FakeOllamaClient(["draft answer", "reviewed answer"])
    chat_settings = SimpleNamespace(debug_enabled=False)

    await bot._generate_response(
        [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "tell me about the recent changes with AI"},
        ],
        [],
        chat_settings,
    )

    review_call = bot.ollama_client.calls[1]
    assert any(
        "tell me about the recent changes with AI" in message["content"]
        for message in review_call
        if message["role"] == "user"
    )


async def test_command_decision_uses_model_output() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(['{"cmd":"research","search":"bowtiedcyber","url":null,"reason":"user asked for investigation"}'])
    bot.context_assembler = ContextAssembler("system prompt", chat_window_size=20)
    profile = UserProfile(chat_id="1", onboarding_complete=True, active_projects=["BowTiedCyber"])

    decision = await bot._decide_command("Do research on bowtiedcyber and see what you find", profile, [])

    assert decision == CommandDecision(cmd="research", search="bowtiedcyber", url=None, reason="user asked for investigation")


async def test_command_decision_falls_back_to_url_research() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(["not json"])
    bot.context_assembler = ContextAssembler("system prompt", chat_window_size=20)

    decision = await bot._decide_command("Tell me about https://example.com", UserProfile(chat_id="1"), [])

    assert decision.cmd == "research"
    assert decision.url == "https://example.com"


async def test_command_decision_prompt_includes_profile_and_recent_context() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(['{"cmd":"chat","search":null,"url":null,"reason":"follow-up planning"}'])
    bot.context_assembler = ContextAssembler("system prompt", chat_window_size=20)
    profile = UserProfile(chat_id="1", onboarding_complete=True, active_projects=["My Agency"], primary_goals=["Grow revenue"])
    conversation = [
        ChatMessage(id="1", role="user", text="I run My Agency"),
        ChatMessage(id="2", role="assistant", text="Got it."),
    ]

    decision = await bot._decide_command("what now?", profile, conversation)

    assert decision.cmd == "chat"
    planner_system_messages = bot.ollama_client.calls[0]
    assert any("My Agency" in message["content"] for message in planner_system_messages)
    assert any("what now?" == message["content"] for message in planner_system_messages if message["role"] == "user")


def test_prompt_summary_keeps_recent_message_previews() -> None:
    summary = LlamaClawBot._summarize_prompt(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user question"},
            {"role": "assistant", "content": "assistant draft"},
        ]
    )

    assert summary["message_count"] == 3
    assert summary["messages"][-1]["preview"] == "assistant draft"


def test_split_text_for_telegram_chunks_long_output() -> None:
    text = ("Section one\n\n" + ("A" * 2500) + "\n\nSection two\n\n" + ("B" * 2500)).strip()

    chunks = LlamaClawBot._split_text_for_telegram(text, limit=3000)

    assert len(chunks) == 2
    assert all(len(chunk) <= 3000 for chunk in chunks)
    assert "Section one" in "".join(chunks)
    assert "Section two" in "".join(chunks)


def test_clean_plain_text_removes_markdown_emphasis() -> None:
    cleaned = LlamaClawBot._clean_plain_text("**Bold**\n## Heading\nPlain text\n\n\nExtra")

    assert "**" not in cleaned
    assert "##" not in cleaned
    assert "Bold" in cleaned
    assert "\n\n\n" not in cleaned


def test_classify_research_intent_detects_news() -> None:
    intent = LlamaClawBot._classify_research_intent(
        "tell me about the recent changes with AI",
        CommandDecision(cmd="research", search="recent AI changes"),
    )

    assert intent == "news"


def test_sanitize_research_queries_removes_bad_templates() -> None:
    queries = LlamaClawBot._sanitize_research_queries(
        ["recent AI changes", "recent AI changes pricing offer", "recent AI changes team founder"],
        "news",
    )

    assert all("pricing offer" not in query for query in queries)
    assert all("team founder" not in query for query in queries)
    assert any("latest news" in query.lower() or "news" in query.lower() for query in queries)


def test_sanitize_research_plan_clears_news_crawls_and_placeholders() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.settings = SimpleNamespace(research_parallel_crawl_workers=4, research_parallel_search_workers=8)
    raw_plan = ResearchPlan(
        intent_type="news",
        search_queries=["AI updates", "AI in [user industry] latest news"],
        crawl_urls=["https://www.science.org"],
        goal="AI updates",
    )

    sanitized = bot._sanitize_research_plan(raw_plan, CommandDecision(cmd="research", search="AI updates"))

    assert sanitized.crawl_urls == []
    assert all("[" not in query for query in sanitized.search_queries)


def test_score_evidence_prefers_recent_named_sources_for_news() -> None:
    strong = ResearchEvidence(
        claim="OpenAI released a new model update",
        source="Reuters",
        date="2026-03-19",
        relevance_score=0.4,
    )
    weak = ResearchEvidence(
        claim="AI has changed many industries",
        source="Generic Blog",
        date="2022-01-01",
        relevance_score=0.4,
    )

    assert LlamaClawBot._score_evidence(strong, "recent changes with AI", "news") > LlamaClawBot._score_evidence(weak, "recent changes with AI", "news")


def test_format_evidence_packet_outputs_plain_ranked_lines() -> None:
    packet = LlamaClawBot._format_evidence_packet(
        [
            ResearchEvidence(
                claim="OpenAI shipped a model update",
                source="Reuters",
                date="2026-03-19",
                relevance_score=0.91,
            )
        ]
    )

    assert "Evidence packet:" in packet
    assert "source=Reuters" in packet


def test_evidence_from_search_results_prefers_snippets() -> None:
    evidence = LlamaClawBot._evidence_from_search_results(
        [
            {
                "title": "Reuters AI News | Latest Headlines",
                "url": "https://www.reuters.com/technology/artificial-intelligence/",
                "snippet": "OpenAI launched a new enterprise feature this week, according to Reuters.",
            }
        ],
        "news",
    )

    assert len(evidence) == 1
    assert "OpenAI launched" in evidence[0].claim
    assert "reuters.com" in evidence[0].source


async def test_build_research_outline_uses_model_output() -> None:
    bot = object.__new__(LlamaClawBot)
    bot.ollama_client = FakeOllamaClient(['{"title":"Modern AI Advancements","sections":["What Changed","Why It Matters"]}'])

    outline = await bot._build_research_outline([{"role": "system", "content": "prompt"}], "tell me about ai")

    assert outline == ResearchOutline(title="Modern AI Advancements", sections=["What Changed", "Why It Matters"])
