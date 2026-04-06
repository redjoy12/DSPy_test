import dspy

from src.config import configure_lm, get_default_model


def test_get_default_model_returns_expected_value():
    model = get_default_model()
    assert model == "openai/gpt-4o-mini"


def test_configure_lm_returns_lm_instance():
    lm = configure_lm(model="openai/gpt-4o-mini")
    assert isinstance(lm, dspy.LM)
    dspy.settings.lm = None


def test_configure_lm_sets_global_dspy_lm():
    lm = configure_lm(model="openai/gpt-4o-mini")
    assert dspy.settings.lm is lm
    dspy.settings.lm = None


def test_configure_lm_respects_custom_params():
    lm = configure_lm(model="openai/gpt-4o", temperature=0.3, max_tokens=500)
    assert lm.model == "openai/gpt-4o"
    assert lm.kwargs["temperature"] == 0.3
    assert lm.kwargs["max_tokens"] == 500
    dspy.settings.lm = None
