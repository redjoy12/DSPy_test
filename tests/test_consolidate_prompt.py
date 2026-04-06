import pytest
import dspy
from dspy.utils import DummyLM

from src.pipelines.consolidate_prompt import (
    ConsolidatePromptSignature,
    PromptConsolidator,
)


class TestConsolidatePromptSignature:
    def test_input_fields(self):
        fields = ConsolidatePromptSignature.input_fields
        assert "raw_prompt" in fields
        assert "original_prompt" in fields
        assert "change_request" in fields
        assert "abstracted_pattern" in fields

    def test_output_fields(self):
        fields = ConsolidatePromptSignature.output_fields
        assert "consolidated_prompt" in fields
        assert "consolidation_notes" in fields


class TestPromptConsolidator:
    def test_forward_returns_consolidated_prompt_and_notes(self):
        """The consolidator returns a (cleaned_prompt, notes) tuple."""
        lm = DummyLM(
            [
                {
                    "reasoning": "Removed scenario specifics and merged duplicate rules.",
                    "consolidated_prompt": "You are a concise assistant.",
                    "consolidation_notes": "Scrubbed product name; merged 2 conciseness rules into 1.",
                }
            ]
        )
        with dspy.context(lm=lm):
            consolidator = PromptConsolidator()
            consolidated, notes = consolidator(
                raw_prompt="You are an assistant. Be concise. Keep responses short. Mention Acme Pro by name.",
                original_prompt="You are an assistant. Be concise.",
                change_request="Make it general",
                abstracted_pattern="Issue: too verbose\nPattern: long answers\nRoot cause: missing brevity rule",
            )

        assert consolidated == "You are a concise assistant."
        assert "merged" in notes.lower() or "scrub" in notes.lower()

    def test_forward_strips_surrounding_whitespace(self):
        lm = DummyLM(
            [
                {
                    "reasoning": "x",
                    "consolidated_prompt": "  cleaned prompt  \n",
                    "consolidation_notes": "  notes here  ",
                }
            ]
        )
        with dspy.context(lm=lm):
            consolidator = PromptConsolidator()
            consolidated, notes = consolidator(
                raw_prompt="raw",
                original_prompt="base",
                change_request="cr",
                abstracted_pattern="",
            )
        assert consolidated == "cleaned prompt"
        assert notes == "notes here"

    def test_forward_raises_on_empty_consolidated_prompt(self):
        """Empty consolidated_prompt must raise instead of silently dropping the user's prompt."""
        lm = DummyLM(
            [
                {
                    "reasoning": "Returned nothing.",
                    "consolidated_prompt": "",
                    "consolidation_notes": "",
                }
            ]
        )
        with dspy.context(lm=lm):
            consolidator = PromptConsolidator()
            with pytest.raises(RuntimeError, match="empty"):
                consolidator(
                    raw_prompt="some prompt",
                    original_prompt="base",
                    change_request="cr",
                    abstracted_pattern="",
                )

    def test_forward_wraps_llm_errors_in_runtime_error(self):
        class ExplodingModule:
            def __call__(self, **kwargs):
                raise ValueError("LLM blew up")

        consolidator = PromptConsolidator(consolidator_module=ExplodingModule())
        with pytest.raises(RuntimeError, match="Consolidation failed"):
            consolidator(
                raw_prompt="x",
                original_prompt="base",
                change_request="cr",
                abstracted_pattern="",
            )

    def test_forward_passes_inputs_through_to_llm(self):
        captured: dict = {}

        class _CapturingModule:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    consolidated_prompt="ok",
                    consolidation_notes="done",
                )

        consolidator = PromptConsolidator(consolidator_module=_CapturingModule())
        consolidator(
            raw_prompt="raw text",
            original_prompt="the baseline prompt",
            change_request="the change request",
            abstracted_pattern="the abstracted pattern",
        )

        assert captured["raw_prompt"] == "raw text"
        assert captured["original_prompt"] == "the baseline prompt"
        assert captured["change_request"] == "the change request"
        assert captured["abstracted_pattern"] == "the abstracted pattern"

    def test_forward_requires_original_prompt(self):
        """original_prompt is required — consolidator has no baseline without it."""
        consolidator = PromptConsolidator()
        with pytest.raises(TypeError):
            # Intentionally omit original_prompt to confirm it's required.
            consolidator(raw_prompt="x", change_request="cr", abstracted_pattern="")

    def test_captures_original_prompt_for_baseline(self):
        """The original prompt is forwarded verbatim so the LLM can use it as the baseline."""
        captured: dict = {}

        class _CapturingModule:
            def __call__(self, **kwargs):
                captured.update(kwargs)
                return dspy.Prediction(
                    consolidated_prompt="final", consolidation_notes="n"
                )

        baseline = (
            "You are a senior code reviewer.\n"
            "1. Be concise.\n"
            "2. Use a direct tone.\n"
        )
        consolidator = PromptConsolidator(consolidator_module=_CapturingModule())
        consolidator(
            raw_prompt=baseline + "3. Also mention product pricing when asked.",
            original_prompt=baseline,
            change_request="handle pricing questions",
            abstracted_pattern="",
        )

        # Baseline must reach the signature byte-for-byte, otherwise the LLM
        # cannot enforce the off-limits rule.
        assert captured["original_prompt"] == baseline

    def test_default_module_is_chain_of_thought(self):
        """When no consolidator_module is injected, defaults to ChainOfThought of the signature."""
        consolidator = PromptConsolidator()
        assert isinstance(consolidator.consolidate, dspy.ChainOfThought)
