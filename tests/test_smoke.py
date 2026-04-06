"""
Integration smoke tests that verify AI behavior with a real LLM.

These tests require a valid OPENAI_API_KEY environment variable and
make actual API calls. They are excluded from the default test suite.

Run with: python -m pytest tests/test_smoke.py -m integration
"""

import pytest
import dspy

from src.config import configure_lm
from src.pipelines.create_prompt import CreatePromptPipeline
from src.pipelines.iterate_prompt import IteratePromptPipeline
from src.evaluation.judge import PromptQualityJudge
from src.evaluation.example_metric import ExampleBasedMetric
from src.store.prompt_store import PromptStore


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_lm(pytestconfig):
    """Configure a real LLM for integration tests."""
    if "integration" not in pytestconfig.getmarkers("integration"):
        pytest.skip("Integration test only")
    configure_lm(model="openai/gpt-4o-mini")


@pytest.fixture
def store(tmp_path):
    return PromptStore(base_dir=tmp_path)


class TestCreatePipelineSmoke:
    def test_produces_relevant_output(self, store):
        pipeline = CreatePromptPipeline(store=store)
        result = pipeline.create_and_save(
            name="smoke_create",
            description="A Python code reviewer that checks for bugs",
            model="openai/gpt-4o-mini",
        )
        assert len(result.prompt_text) > 20
        lower = result.prompt_text.lower()
        assert any(word in lower for word in ["code", "review", "python", "bug"])

    def test_different_descriptions_produce_different_prompts(self, store):
        pipeline = CreatePromptPipeline(store=store)
        v1 = pipeline.create_and_save(
            name="smoke_a",
            description="A friendly customer support chatbot",
            model="openai/gpt-4o-mini",
        )
        v2 = pipeline.create_and_save(
            name="smoke_b",
            description="A strict math tutor",
            model="openai/gpt-4o-mini",
        )
        assert v1.prompt_text != v2.prompt_text


class TestIteratePipelineSmoke:
    def test_iteration_modifies_prompt(self, store):
        pipeline = IteratePromptPipeline(store=store)
        pipeline.create_and_save(
            "smoke_iterate",
            "A helpful writing assistant",
            model="openai/gpt-4o-mini",
        )
        original = store.load_latest("smoke_iterate")
        pipeline.iterate_and_save(
            "smoke_iterate",
            change_request="Make it more concise and direct",
            model="openai/gpt-4o-mini",
        )
        iterated = store.load_latest("smoke_iterate")
        assert original.prompt_text != iterated.prompt_text


class TestJudgeSmoke:
    def test_evaluate_quality_returns_valid_score(self):
        judge = PromptQualityJudge()
        score, feedback = judge.evaluate_quality(
            prompt_text="You are a helpful AI assistant that answers questions clearly.",
            description="A helpful Q&A assistant",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert len(feedback) > 0

    def test_bad_prompt_scores_lower_than_good_prompt(self):
        judge = PromptQualityJudge()
        good_score, _ = judge.evaluate_quality(
            prompt_text=(
                "You are an expert Python developer. Review code for bugs, "
                "style issues, and performance problems. Provide clear explanations."
            ),
            description="A Python code reviewer",
        )
        bad_score, _ = judge.evaluate_quality(
            prompt_text="Do stuff with code maybe",
            description="A Python code reviewer",
        )
        assert good_score > bad_score


class TestExampleBasedMetricSmoke:
    def test_evaluate_returns_positive_score(self):
        metric = ExampleBasedMetric()
        examples = [
            {"input": "What is 2+2?", "expected_output": "4"},
            {"input": "What is the capital of France?", "expected_output": "Paris"},
        ]
        score = metric.evaluate(
            "You are a helpful assistant that answers questions accurately.",
            examples,
        )
        assert isinstance(score, float)
        assert score > 0.0
