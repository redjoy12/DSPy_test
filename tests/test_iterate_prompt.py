import pytest
import dspy
from dspy.utils import DummyLM

from src.pipelines.iterate_prompt import IteratePromptSignature, IteratePromptPipeline
from src.pipelines.create_prompt import CreatePromptPipeline
from src.store.prompt_store import PromptStore


class _PassthroughConsolidator:
    """Test stub: returns the raw prompt unchanged with empty notes.

    Injected into iterate_and_save tests so the consolidation post-pass
    doesn't consume DummyLM completions or alter the prompt under test.
    Tests that specifically verify the consolidation integration use a
    different stub or the real PromptConsolidator with a primed DummyLM.
    """

    def __call__(
        self, raw_prompt, original_prompt="", change_request="", abstracted_pattern=""
    ):
        return raw_prompt, ""


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
        lm = DummyLM(
            [
                {
                    "reasoning": "Made it more concise.",
                    "improved_prompt": "You are a concise assistant.",
                    "changes_made": "Removed verbose instructions.",
                }
            ]
        )
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
        lm = DummyLM(
            [
                {
                    "reasoning": "Addressed failures.",
                    "improved_prompt": "Improved prompt.",
                    "changes_made": "Fixed edge cases.",
                }
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline(
                current_prompt="Old prompt",
                change_request="Handle errors",
                failing_examples="Input: 'error' -> Expected: 'graceful response'",
            )
        assert result.improved_prompt == "Improved prompt."

    def test_forward_defaults_to_empty_failing_examples(self):
        lm = DummyLM(
            [
                {
                    "reasoning": "x.",
                    "improved_prompt": "Better.",
                    "changes_made": "Improved.",
                }
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline(current_prompt="Old", change_request="Improve tone")
        assert result.improved_prompt == "Better."


class TestIterateAndSave:
    def _create_v1(self, store):
        """Helper: create an initial prompt version in the store."""
        lm = DummyLM(
            [
                {"reasoning": "Initial.", "prompt_text": "You are a coding assistant."},
                {
                    "reasoning": "Good.",
                    "quality_score": "0.85",
                    "feedback": "Good start.",
                },
            ]
        )
        with dspy.context(lm=lm):
            cp = CreatePromptPipeline(store=store)
            return cp.create_and_save(
                name="test_prompt", description="Coding assistant"
            )

    def test_iterates_from_stored_prompt(self, store):
        """Loads the latest version from store, iterates, and saves v2."""
        self._create_v1(store)

        lm = DummyLM(
            [
                {
                    "reasoning": "Added error handling.",
                    "improved_prompt": "You are a coding assistant that explains errors clearly.",
                    "changes_made": "Added error explanation behavior.",
                },
                {
                    "reasoning": "Better.",
                    "improvement_score": "0.92",
                    "feedback": "Good improvement.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_PassthroughConsolidator()
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt", change_request="Add error handling"
            )

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
        lm = DummyLM(
            [
                {
                    "reasoning": "Improved.",
                    "improved_prompt": "Better prompt.",
                    "changes_made": "Improved clarity.",
                },
                {
                    "reasoning": "OK.",
                    "improvement_score": "0.85",
                    "feedback": "Decent.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_PassthroughConsolidator()
            )
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
            lm = DummyLM(
                [
                    {
                        "reasoning": f"Iteration {i}.",
                        "improved_prompt": f"Prompt version {i}.",
                        "changes_made": f"Change {i}.",
                    },
                    {
                        "reasoning": "x.",
                        "improvement_score": "0.9",
                        "feedback": "Good.",
                    },
                ]
            )
            with dspy.context(lm=lm):
                pipeline = IteratePromptPipeline(
                    store=store, consolidator=_PassthroughConsolidator()
                )
                pipeline.iterate_and_save(
                    name="test_prompt", change_request=f"Improvement {i}"
                )

        versions = store.list_versions("test_prompt")
        assert versions == [1, 2, 3, 4]

        latest = store.load_latest("test_prompt")
        assert latest.version == 4
        assert latest.parent_version == 3

    def test_iterates_with_structured_examples(self, store):
        """Structured examples should be serialized, abstracted, then passed to iteration.
        The improved prompt should address the general pattern, not the specific scenario."""
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

        lm = DummyLM(
            [
                {
                    "reasoning": "Extracting pattern.",
                    "issue": "Bot lacks multi-turn context",
                    "pattern": "User asks follow-up questions",
                    "root_cause": "No context tracking",
                },
                {"reasoning": "Good.", "improvement_score": "0.9", "feedback": "Nice."},
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                generate_module=_CapturingGenerate(),
                store=store,
                consolidator=_PassthroughConsolidator(),
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Handle multi-turn product questions",
                structured_examples=structured,
            )

        assert "Issue:" in captured["failing_examples"]
        assert "Pattern:" in captured["failing_examples"]
        assert "Root cause:" in captured["failing_examples"]
        assert "--- Example 1 ---" not in captured["failing_examples"]

        assert v2.structured_examples == structured
        loaded = store.load("test_prompt", 2)
        assert loaded.structured_examples == structured
        assert loaded.abstracted_patterns is not None
        assert "Issue:" in loaded.abstracted_patterns

    def test_structured_examples_combine_with_failing_examples_text(self, store):
        """Passing both a failing_examples string and structured examples
        results in combined abstraction being passed to iteration."""
        self._create_v1(store)

        captured: dict = {}

        class _CapturingGenerate:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    improved_prompt="Improved prompt with better handling for user queries.",
                    changes_made="Merged inputs.",
                )

        lm = DummyLM(
            [
                {
                    "reasoning": "Extracting.",
                    "issue": "General issue with handling",
                    "pattern": "User provides input with expectations",
                    "root_cause": "Missing guidance for specific queries",
                },
                {"reasoning": "Good.", "improvement_score": "0.9", "feedback": "Fine."},
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                generate_module=_CapturingGenerate(),
                store=store,
                consolidator=_PassthroughConsolidator(),
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
        assert "Issue:" in text
        assert "Pattern:" in text
        assert "Root cause:" in text

    def test_iterate_and_save_rejects_low_quality_with_min_score(self, store):
        """When min_score is set and the iteration scores below it, ValueError is raised."""
        self._create_v1(store)

        lm = DummyLM(
            [
                {
                    "reasoning": "Weak iteration.",
                    "improved_prompt": "A bad prompt.",
                    "changes_made": "Made it worse.",
                },
                {
                    "reasoning": "Poor.",
                    "improvement_score": "0.3",
                    "feedback": "Low quality.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_PassthroughConsolidator()
            )
            with pytest.raises(ValueError, match="below minimum threshold"):
                pipeline.iterate_and_save(
                    name="test_prompt",
                    change_request="Make it worse",
                    min_score=0.5,
                )

    def test_iterate_and_save_validation_failure_raises_error(self, store):
        """When validation detects literal copying, ValueError is raised."""
        self._create_v1(store)

        class FailingPatternExtractor:
            def __call__(self, **kwargs):
                raise RuntimeError("Pattern extraction failed")

        lm = DummyLM(
            [
                {
                    "reasoning": "Generating improved prompt.",
                    "improved_prompt": "Human: What products do you have? -> Assistant: We have X, Y, and Z.",
                    "changes_made": "Added product guidance.",
                },
                {
                    "reasoning": "Good.",
                    "improvement_score": "0.9",
                    "feedback": "Nice.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store,
                pattern_extractor=FailingPatternExtractor(),
                consolidator=_PassthroughConsolidator(),
            )
            with pytest.raises(ValueError, match="Validation failed"):
                pipeline.iterate_and_save(
                    name="test_prompt",
                    change_request="Handle questions",
                    failing_examples="Human: What products do you have? -> Assistant: We have X, Y, and Z",
                )

    def test_iterate_and_save_pattern_extractor_fallback_on_error(self, store):
        """When PatternExtractor fails, pipeline falls back to original failing_examples."""
        self._create_v1(store)

        class FailingPatternExtractor:
            def __call__(self, **kwargs):
                raise RuntimeError("Pattern extraction failed")

        captured: dict = {}

        class _CapturingGenerate:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    improved_prompt="Improved prompt.",
                    changes_made="Improved.",
                )

        lm = DummyLM(
            [
                {"reasoning": "Good.", "improvement_score": "0.9", "feedback": "Nice."},
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store,
                generate_module=_CapturingGenerate(),
                pattern_extractor=FailingPatternExtractor(),
                consolidator=_PassthroughConsolidator(),
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Improve",
                failing_examples="Input: test -> Output: result",
            )

        assert v2 is not None
        assert "Input: test -> Output: result" in captured["failing_examples"]

    def test_iterate_and_save_interactive_accept(self, store):
        """When interactive=True and user accepts, validation passes."""
        self._create_v1(store)

        lm = DummyLM(
            [
                {
                    "reasoning": "Made a change.",
                    "improved_prompt": "You are a coding assistant with error handling.",
                    "changes_made": "Added error handling.",
                },
                {
                    "reasoning": "Good.",
                    "improvement_score": "0.85",
                    "feedback": "Good.",
                },
            ]
        )
        response_log = []

        def mock_input(prompt):
            response_log.append(prompt)
            return "a"

        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_PassthroughConsolidator()
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Add error handling",
                interactive=True,
                prompt_func=mock_input,
            )

        assert v2 is not None
        assert v2.version == 2

    def test_iterate_and_save_interactive_retry(self, store):
        """When interactive=True and user retries, ValueError is raised."""
        self._create_v1(store)

        lm = DummyLM(
            [
                {
                    "reasoning": "Extracting pattern.",
                    "issue": "General issue",
                    "pattern": "User provides input",
                    "root_cause": "Missing guidance",
                },
                {
                    "reasoning": "Made a change.",
                    "improved_prompt": "Input: 'error message' -> Output: 'oops something went wrong'",
                    "changes_made": "Changed it.",
                },
                {
                    "reasoning": "OK.",
                    "improvement_score": "0.85",
                    "feedback": "Good.",
                },
            ]
        )

        def mock_input(prompt):
            return "r"

        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_PassthroughConsolidator()
            )
            with pytest.raises(ValueError, match="User requested retry"):
                pipeline.iterate_and_save(
                    name="test_prompt",
                    change_request="Add error handling",
                    failing_examples="Input: 'error message' -> Output: 'oops something went wrong'",
                    interactive=True,
                    prompt_func=mock_input,
                )

    def test_consolidator_output_is_what_gets_saved(self, store):
        """The consolidated prompt (not the raw improved_prompt) becomes prompt_text."""
        self._create_v1(store)

        class _RewritingConsolidator:
            def __call__(
                self,
                raw_prompt,
                original_prompt="",
                change_request="",
                abstracted_pattern="",
            ):
                return ("CONSOLIDATED VERSION", "merged 2 rules; scrubbed 1 specific name")

        lm = DummyLM(
            [
                {
                    "reasoning": "Iterating.",
                    "improved_prompt": "RAW LLM OUTPUT mentioning Acme Pro",
                    "changes_made": "Added a rule.",
                },
                {
                    "reasoning": "OK.",
                    "improvement_score": "0.9",
                    "feedback": "Good.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_RewritingConsolidator()
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt", change_request="Add a rule"
            )

        assert v2.prompt_text == "CONSOLIDATED VERSION"
        assert "Acme Pro" not in v2.prompt_text
        # Consolidation notes flow into changes_made for transparency.
        assert "Consolidation" in v2.changes_made
        assert "merged 2 rules" in v2.changes_made

        # Verify the persisted version on disk reflects the consolidated text too.
        loaded = store.load("test_prompt", 2)
        assert loaded.prompt_text == "CONSOLIDATED VERSION"

    def test_consolidator_failure_falls_back_to_raw_prompt(self, store, caplog):
        """When the consolidator raises, the raw improved_prompt is saved and a warning is logged."""
        import logging

        self._create_v1(store)

        class _ExplodingConsolidator:
            def __call__(
                self,
                raw_prompt,
                original_prompt="",
                change_request="",
                abstracted_pattern="",
            ):
                raise RuntimeError("consolidator boom")

        lm = DummyLM(
            [
                {
                    "reasoning": "Iterating.",
                    "improved_prompt": "Raw improved prompt about general behaviors.",
                    "changes_made": "Added rules.",
                },
                {
                    "reasoning": "OK.",
                    "improvement_score": "0.9",
                    "feedback": "Good.",
                },
            ]
        )
        with dspy.context(lm=lm), caplog.at_level(logging.WARNING):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_ExplodingConsolidator()
            )
            v2 = pipeline.iterate_and_save(
                name="test_prompt", change_request="Improve"
            )

        assert v2.prompt_text == "Raw improved prompt about general behaviors."
        assert any("Consolidation failed" in r.message for r in caplog.records)
        # No consolidation section in changes_made when notes are empty.
        assert "Consolidation (scrub + merge)" not in v2.changes_made

    def test_consolidator_runs_after_iterate_with_correct_inputs(self, store):
        """The consolidator receives the iterate output, original_prompt (baseline), change_request, and abstracted pattern."""
        self._create_v1(store)

        captured: dict = {}

        class _CapturingConsolidator:
            def __call__(
                self,
                raw_prompt,
                original_prompt="",
                change_request="",
                abstracted_pattern="",
            ):
                captured["raw_prompt"] = raw_prompt
                captured["original_prompt"] = original_prompt
                captured["change_request"] = change_request
                captured["abstracted_pattern"] = abstracted_pattern
                return ("a clean consolidated prompt", "merged notes")

        structured = [
            {
                "messages": [{"role": "human", "content": "what?"}],
                "unsatisfactory_output": "I don't know",
            }
        ]

        lm = DummyLM(
            [
                {
                    "reasoning": "Extracting.",
                    "issue": "Bot lacks context",
                    "pattern": "User asks unanswered questions",
                    "root_cause": "Missing fallback rule",
                },
                {
                    "reasoning": "Iterating.",
                    "improved_prompt": "RAW iterate output",
                    "changes_made": "Added fallback.",
                },
                {
                    "reasoning": "OK.",
                    "improvement_score": "0.9",
                    "feedback": "Good.",
                },
            ]
        )
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline(
                store=store, consolidator=_CapturingConsolidator()
            )
            pipeline.iterate_and_save(
                name="test_prompt",
                change_request="Handle unanswerable questions",
                structured_examples=structured,
            )

        assert captured["raw_prompt"] == "RAW iterate output"
        # The baseline fed to the consolidator must be the v1 prompt text
        # loaded from the store — this is the regression guard ensuring
        # consolidation is scoped to the delta, not the whole prompt.
        assert captured["original_prompt"] == "You are a coding assistant."
        assert captured["change_request"] == "Handle unanswerable questions"
        assert "Issue:" in captured["abstracted_pattern"]
        assert "Pattern:" in captured["abstracted_pattern"]
        assert "Root cause:" in captured["abstracted_pattern"]

    def test_detect_model_no_lm_configured(self):
        """_detect_model returns 'unknown' when no LM is configured."""
        pipeline = IteratePromptPipeline()
        result = pipeline._detect_model()
        assert result == "unknown"

    def test_detect_model_with_lm(self):
        """_detect_model returns the model name when LM is configured."""
        lm = DummyLM([])
        with dspy.context(lm=lm):
            pipeline = IteratePromptPipeline()
            result = pipeline._detect_model()
        assert "dummy" in result.lower() or result == "unknown"
