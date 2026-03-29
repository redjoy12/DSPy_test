import pytest
import dspy
from dspy.utils import DummyLM

from src.pipelines.create_prompt import CreatePromptSignature, CreatePromptPipeline
from src.store.prompt_store import PromptStore


class TestCreatePromptSignature:
    def test_input_fields(self):
        assert "description" in CreatePromptSignature.input_fields
        assert "context" in CreatePromptSignature.input_fields

    def test_output_fields(self):
        assert "prompt_text" in CreatePromptSignature.output_fields
        assert "reasoning" in CreatePromptSignature.output_fields


class TestCreatePromptPipeline:
    def test_forward_produces_prompt_text(self):
        """The real DSPy pipeline generates a prompt_text from a description."""
        lm = DummyLM([{
            "reasoning": "Designed for customer support use case.",
            "prompt_text": "You are a friendly customer support agent.",
        }])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline()
            result = pipeline(description="A customer support chatbot")
        assert result.prompt_text == "You are a friendly customer support agent."
        assert result.reasoning == "Designed for customer support use case."

    def test_forward_with_context(self):
        """Context is passed through the real DSPy pipeline."""
        lm = DummyLM([{
            "reasoning": "Formal tone for enterprise.",
            "prompt_text": "You are a professional email classifier.",
        }])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline()
            result = pipeline(description="Email classifier", context="Enterprise SaaS product")
        assert result.prompt_text == "You are a professional email classifier."

    def test_forward_defaults_to_empty_context(self):
        """When no context is given, empty string is used (not None)."""
        lm = DummyLM([{"reasoning": "Basic.", "prompt_text": "Test prompt."}])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline()
            # This should not raise — empty context is valid
            result = pipeline(description="Code reviewer")
        assert result.prompt_text == "Test prompt."


class TestCreateAndSave:
    def test_stores_versioned_prompt_to_disk(self, store):
        """Full pipeline: generate -> judge -> save to real store."""
        lm = DummyLM([
            {"reasoning": "Role-based prompt.", "prompt_text": "You are a code reviewer."},
            {"reasoning": "Good structure.", "quality_score": "0.88", "feedback": "Well structured."},
        ])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline(store=store)
            result = pipeline.create_and_save(name="code_reviewer", description="A code review assistant")

        assert result.version == 1
        assert result.parent_version is None
        assert result.prompt_text == "You are a code reviewer."
        assert result.quality_score == 0.88
        assert result.judge_feedback == "Well structured."
        assert result.pipeline == "create"

        # Verify it was actually persisted to the filesystem
        loaded = store.load("code_reviewer", 1)
        assert loaded.prompt_text == result.prompt_text
        assert loaded.quality_score == result.quality_score

    def test_increments_version_on_second_create(self, store):
        """Creating the same prompt name twice produces version 2."""
        lm = DummyLM([
            {"reasoning": "v1.", "prompt_text": "Version 1 prompt."},
            {"reasoning": "j1.", "quality_score": "0.8", "feedback": "OK."},
            {"reasoning": "v2.", "prompt_text": "Version 2 prompt."},
            {"reasoning": "j2.", "quality_score": "0.9", "feedback": "Better."},
        ])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline(store=store)
            v1 = pipeline.create_and_save(name="my_prompt", description="Test")
            v2 = pipeline.create_and_save(name="my_prompt", description="Test improved")

        assert v1.version == 1
        assert v2.version == 2
        assert store.list_versions("my_prompt") == [1, 2]

    def test_create_and_save_rejects_low_quality_with_min_score(self, store):
        lm = DummyLM([
            {"reasoning": "Test reasoning", "prompt_text": "Bad prompt"},
            {"reasoning": "Low quality", "quality_score": "0.3", "feedback": "Poor"},
        ])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline(store=store)
            with pytest.raises(ValueError, match="below minimum threshold"):
                pipeline.create_and_save("test", "desc", min_score=0.5)

    def test_quality_score_from_judge_is_stored(self, store):
        """The judge's score is what gets stored, not a hardcoded value."""
        lm = DummyLM([
            {"reasoning": "x.", "prompt_text": "A prompt."},
            {"reasoning": "x.", "quality_score": "0.42", "feedback": "Mediocre."},
        ])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline(store=store)
            result = pipeline.create_and_save(name="scored", description="Test")

        assert result.quality_score == 0.42
        loaded = store.load("scored", 1)
        assert loaded.quality_score == 0.42
