import pytest
from unittest.mock import patch

from src.config import configure_lm, get_default_model


def test_get_default_model_returns_string():
    model = get_default_model()
    assert isinstance(model, str)
    assert "openai" in model or "gpt" in model


def test_configure_lm_sets_dspy_config():
    """Verify configure_lm calls dspy.configure with an LM instance."""
    with patch("src.config.dspy") as mock_dspy:
        mock_dspy.LM.return_value = "fake_lm"
        configure_lm(model="openai/gpt-4o-mini")
        mock_dspy.LM.assert_called_once_with("openai/gpt-4o-mini", temperature=0.7, max_tokens=2000)
        mock_dspy.configure.assert_called_once_with(lm="fake_lm")


def test_configure_lm_custom_params():
    """Verify custom temperature and max_tokens are passed through."""
    with patch("src.config.dspy") as mock_dspy:
        mock_dspy.LM.return_value = "fake_lm"
        configure_lm(model="openai/gpt-4o", temperature=0.3, max_tokens=500)
        mock_dspy.LM.assert_called_once_with("openai/gpt-4o", temperature=0.3, max_tokens=500)
