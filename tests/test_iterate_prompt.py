import pytest
import dspy
from dspy.utils import DummyLM

from src.pipelines.iterate_prompt import IteratePromptSignature, IteratePromptPipeline
from src.pipelines.create_prompt import CreatePromptPipeline
from src.store.prompt_store import PromptStore


class TestIteratePromptSignature:
    def test_input_fields(self):
        fields = IteratePromptSignature.input_fields
        assert "current_prompt" in fields
        assert "change_request" in fields
        assert "failing_examples" in fields

    def test_output_fields(self):
        fields = IteratePromptSignature.output_fields
        assert "improved_prompt" in fields
        assert "changes_made" in fields


class TestIteratePromptPipeline:
    def test_forward_produces_improved_prompt(self):
        """The real DSPy pipeline returns an improved prompt and changes summary."""
        lm = DummyLM([{
            "reasoning": "Made it more concise.",
            "improved_prompt": "You are a concise assistant.",
            "changes_made": "Removed verbose instructions.",
        }])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline(
                current_prompt="You are a very verbose assistant that says too much.",
                change_request="Make it more concise",
            )
        assert result.improved_prompt == "You are a concise assistant."
        assert result.changes_made == "Removed verbose instructions."

    def test_forward_with_failing_examples(self):
        """Failing examples are accepted by the real pipeline."""
        lm = DummyLM([{
            "reasoning": "Addressed failures.",
            "improved_prompt": "Improved prompt.",
            "changes_made": "Fixed edge cases.",
        }])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline(
                current_prompt="Old prompt",
                change_request="Handle errors",
                failing_examples="Input: 'error' -> Expected: 'graceful response'",
            )
        assert result.improved_prompt == "Improved prompt."

    def test_forward_defaults_to_empty_failing_examples(self):
        lm = DummyLM([{
            "reasoning": "x.",
            "improved_prompt": "Better.",
            "changes_made": "Improved.",
        }])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline(current_prompt="Old", change_request="Improve tone")
        assert result.improved_prompt == "Better."


class TestIterateAndSave:
    def _create_v1(self, store):
        """Helper: create an initial prompt version in the store."""
        lm = DummyLM([
            {"reasoning": "Initial.", "prompt_text": "You are a coding assistant."},
            {"reasoning": "Good.", "quality_score": "0.85", "feedback": "Good start."},
        ])
        with dspy.context(lm=lm):
            cp = CreatePromptPipeline(store=store)
            return cp.create_and_save(name="test_prompt", description="Coding assistant")

    def test_iterates_from_stored_prompt(self, store):
        """Loads the latest version from store, iterates, and saves v2."""
        self._create_v1(store)

        lm = DummyLM([
            {
                "reasoning": "Added error handling.",
                "improved_prompt": "You are a coding assistant that explains errors clearly.",
                "changes_made": "Added error explanation behavior.",
            },
            {"reasoning": "Better.", "improvement_score": "0.92", "feedback": "Good improvement."},
        ])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(store=store)
            v2 = pipeline.iterate_and_save(name="test_prompt", change_request="Add error handling")

        assert v2.version == 2
        assert v2.parent_version == 1
        assert v2.pipeline == "iterate"
        assert v2.change_request == "Add error handling"
        assert "error" in v2.prompt_text.lower()

        # Verify lineage on disk
        loaded_v1 = store.load("test_prompt", 1)
        loaded_v2 = store.load("test_prompt", 2)
        assert loaded_v2.parent_version == loaded_v1.version

    def test_iterates_with_explicit_prompt(self, store):
        """When passing current_prompt directly, skips store lookup."""
        lm = DummyLM([
            {
                "reasoning": "Improved.",
                "improved_prompt": "Better prompt.",
                "changes_made": "Improved clarity.",
            },
            {"reasoning": "OK.", "improvement_score": "0.85", "feedback": "Decent."},
        ])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(store=store)
            result = pipeline.iterate_and_save(
                name="new_prompt",
                change_request="Improve it",
                current_prompt="Original prompt.",
                description="A test prompt",
            )

        assert result.version == 1
        assert result.parent_version is None  # no prior version exists
        assert result.prompt_text == "Better prompt."

    def test_builds_version_lineage_across_iterations(self, store):
        """Multiple iterations create a proper version chain."""
        self._create_v1(store)

        for i in range(2, 5):
            lm = DummyLM([
                {
                    "reasoning": f"Iteration {i}.",
                    "improved_prompt": f"Prompt version {i}.",
                    "changes_made": f"Change {i}.",
                },
                {"reasoning": "x.", "improvement_score": "0.9", "feedback": "Good."},
            ])
            with dspy.context(lm=lm):
                pipeline = IteratePromptPipeline(store=store)
                pipeline.iterate_and_save(name="test_prompt", change_request=f"Improvement {i}")

        versions = store.list_versions("test_prompt")
        assert versions == [1, 2, 3, 4]

        latest = store.load_latest("test_prompt")
        assert latest.version == 4
        assert latest.parent_version == 3
