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
                raw_prompt="raw", change_request="cr", abstracted_pattern=""
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
                    change_request="cr",
                    abstracted_pattern="",
                )

    def test_forward_wraps_llm_errors_in_runtime_error(self):
        class ExplodingModule:
            def __call__(self, **kwargs):
                raise ValueError("LLM blew up")

        consolidator = PromptConsolidator(consolidator_module=ExplodingModule())
        with pytest.raises(RuntimeError, match="Consolidation failed"):
            consolidator(raw_prompt="x", change_request="cr", abstracted_pattern="")

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
            change_request="the change request",
            abstracted_pattern="the abstracted pattern",
        )

        assert captured["raw_prompt"] == "raw text"
        assert captured["change_request"] == "the change request"
        assert captured["abstracted_pattern"] == "the abstracted pattern"

    def test_default_module_is_chain_of_thought(self):
        """When no consolidator_module is injected, defaults to ChainOfThought of the signature."""
        consolidator = PromptConsolidator()
        assert isinstance(consolidator.consolidate, dspy.ChainOfThought)
