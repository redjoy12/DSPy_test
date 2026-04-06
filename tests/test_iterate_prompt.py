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

    def test_iterates_with_structured_examples(self, store):
        """Structured examples should be serialized into failing_examples and
        stored verbatim on the new version."""
        self._create_v1(store)

        captured: dict = {}

        class _CapturingGenerate:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    improved_prompt="Improved with multi-turn insight.",
                    changes_made="Addressed the multi-turn failure.",
                )

        structured = [
            {
                "messages": [
                    {"role": "human", "content": "What products?"},
                    {"role": "assistant", "content": "X, Y, Z"},
                    {"role": "human", "content": "Tell me about X"},
                ],
                "unsatisfactory_output": "I don't know",
            }
        ]

        lm = DummyLM([
            {"reasoning": "Good.", "improvement_score": "0.9", "feedback": "Nice."},
        ])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                generate_module=_CapturingGenerate(),
                store=store,
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Handle multi-turn product questions",
                structured_examples=structured,
            )

        # Structured examples were serialized into failing_examples text.
        assert "--- Example 1 ---" in captured["failing_examples"]
        assert "Human: What products?" in captured["failing_examples"]
        assert "Unsatisfactory Output: I don't know" in captured["failing_examples"]

        # Structured examples are stored verbatim on the new version.
        assert v2.structured_examples == structured
        # And persisted to disk.
        loaded = store.load("test_prompt", 2)
        assert loaded.structured_examples == structured

    def test_structured_examples_combine_with_failing_examples_text(self, store):
        """Passing both a failing_examples string and structured examples
        concatenates them for the signature."""
        self._create_v1(store)

        captured: dict = {}

        class _CapturingGenerate:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    improved_prompt="Improved.",
                    changes_made="Merged inputs.",
                )

        lm = DummyLM([
            {"reasoning": "Good.", "improvement_score": "0.9", "feedback": "Fine."},
        ])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                generate_module=_CapturingGenerate(),
                store=store,
            )
            pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Improve",
                failing_examples="Input: foo -> Expected: bar",
                structured_examples=[
                    {
                        "messages": [{"role": "human", "content": "hi"}],
                        "unsatisfactory_output": "oops",
                    }
                ],
            )

        text = captured["failing_examples"]
        assert "Input: foo -> Expected: bar" in text
        assert "--- Example 1 ---" in text
        assert "Human: hi" in text

    def test_iterate_and_save_rejects_low_quality_with_min_score(self, store):
        """When min_score is set and the iteration scores below it, ValueError is raised."""
        self._create_v1(store)

        lm = DummyLM([
            {
                "reasoning": "Weak iteration.",
                "improved_prompt": "A bad prompt.",
                "changes_made": "Made it worse.",
            },
            {"reasoning": "Poor.", "improvement_score": "0.3", "feedback": "Low quality."},
        ])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(store=store)
            with pytest.raises(ValueError, match="below minimum threshold"):
                pipeline.iterate_and_save(
                    name="test_prompt",
                    change_request="Make it worse",
                    min_score=0.5,
                )
