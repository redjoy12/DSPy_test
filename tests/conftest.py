import pytest
import dspy
from dspy.utils import DummyLM

from src.store.prompt_store import PromptStore


@pytest.fixture
def store(tmp_path):
    """Real file-backed prompt store using a temporary directory."""
    return PromptStore(base_dir=tmp_path)


@pytest.fixture
def dummy_create_and_judge():
    """DummyLM pre-loaded with responses for a create pipeline + quality judge call."""
    return DummyLM([
        {"reasoning": "Designed a role-based prompt.", "prompt_text": "You are a helpful coding assistant."},
        {"reasoning": "Good structure.", "quality_score": "0.88", "feedback": "Well structured prompt."},
    ])


@pytest.fixture
def dummy_iterate_and_judge():
    """DummyLM pre-loaded with responses for an iterate pipeline + comparison judge call."""
    return DummyLM([
        {
            "reasoning": "Added error handling behavior.",
            "improved_prompt": "You are a helpful coding assistant. When errors occur, explain them clearly.",
            "changes_made": "Added error explanation instructions.",
        },
        {"reasoning": "Good improvement.", "improvement_score": "0.92", "feedback": "Addresses the change request."},
    ])
