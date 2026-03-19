from app.models import OnboardingState, UserProfile
from app.services import OnboardingService


def test_onboarding_service_applies_answers() -> None:
    profile = UserProfile(chat_id="1")
    state = OnboardingState(chat_id="1", started=True)

    answers = [
        "Evan",
        "Build LlamaClaw, grow the business",
        "Telegram bot; ad funnel",
        "Johnny is going to college soon",
        "Direct, tactical, concise",
        "Avoid fluff",
        "Quick summaries and source links",
    ]

    next_question = None
    for answer in answers:
        profile, state, next_question = OnboardingService.record_answer(state, profile, answer)

    assert next_question is None
    assert state.completed is True
    assert profile.onboarding_complete is True
    assert profile.preferred_name == "Evan"
    assert "Build LlamaClaw" in profile.primary_goals
    assert "Avoid fluff" in profile.boundaries
