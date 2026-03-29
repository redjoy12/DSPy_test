import pytest
import dspy
from dspy.utils import DummyLM

from src.evaluation.judge import (
    PromptQualityJudge,
    PromptQualitySignature,
    PromptComparisonSignature,
    make_quality_metric,
    make_comparison_metric,
    prompt_quality_metric,
    prompt_comparison_metric,
    QUALITY_THRESHOLD,
)


class TestPromptQualitySignature:
    def test_input_fields(self):
        assert "prompt_text" in PromptQualitySignature.input_fields
        assert "original_description" in PromptQualitySignature.input_fields

    def test_output_fields(self):
        assert "quality_score" in PromptQualitySignature.output_fields
        assert "feedback" in PromptQualitySignature.output_fields


class TestPromptComparisonSignature:
    def test_input_fields(self):
        fields = PromptComparisonSignature.input_fields
        assert "original_prompt" in fields
        assert "improved_prompt" in fields
        assert "change_request" in fields

    def test_output_fields(self):
        fields = PromptComparisonSignature.output_fields
        assert "improvement_score" in fields
        assert "feedback" in fields


class TestEvaluateQuality:
    def test_returns_score_and_feedback(self):
        lm = DummyLM([{"reasoning": "Analysis.", "quality_score": "0.85", "feedback": "Clear prompt."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, feedback = judge.evaluate_quality("You are helpful.", "A chatbot")
        assert score == 0.85
        assert feedback == "Clear prompt."

    def test_clamps_score_above_one(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "1.5", "feedback": "Over-scored."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_quality("Test", "Test")
        assert score == 1.0

    def test_clamps_score_below_zero(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "-0.3", "feedback": "Under-scored."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_quality("Test", "Test")
        assert score == 0.0

    def test_boundary_score_zero(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "0.0", "feedback": "Lowest."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_quality("Test", "Test")
        assert score == 0.0

    def test_boundary_score_one(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "1.0", "feedback": "Highest."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_quality("Test", "Test")
        assert score == 1.0


class TestEvaluateComparison:
    def test_returns_score_and_feedback(self):
        lm = DummyLM([{"reasoning": "Analysis.", "improvement_score": "0.9", "feedback": "Good improvement."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, feedback = judge.evaluate_comparison(
                original_prompt="Old prompt",
                improved_prompt="Better prompt",
                change_request="Improve clarity",
            )
        assert score == 0.9
        assert feedback == "Good improvement."

    def test_clamps_comparison_score_above_one(self):
        lm = DummyLM([{"reasoning": "x", "improvement_score": "2.0", "feedback": "Over."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_comparison("Old", "New", "Fix it")
        assert score == 1.0

    def test_clamps_comparison_score_below_zero(self):
        lm = DummyLM([{"reasoning": "x", "improvement_score": "-1.0", "feedback": "Under."}])
        with dspy.context(lm=lm):
            judge = PromptQualityJudge()
            score, _ = judge.evaluate_comparison("Old", "New", "Fix it")
        assert score == 0.0


class TestMakeQualityMetric:
    def test_returns_float_score(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "0.85", "feedback": "Good."}])
        with dspy.context(lm=lm):
            metric = make_quality_metric()
            example = dspy.Example(description="A chatbot").with_inputs("description")
            pred = dspy.Example(prompt_text="You are helpful.")
            score = metric(example, pred)
        assert score == 0.85

    def test_returns_bool_when_trace_provided(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "0.85", "feedback": "Good."}])
        with dspy.context(lm=lm):
            metric = make_quality_metric()
            example = dspy.Example(description="A chatbot").with_inputs("description")
            pred = dspy.Example(prompt_text="You are helpful.")
            result = metric(example, pred, trace="some_trace")
        assert result is True  # 0.85 >= QUALITY_THRESHOLD (0.7)

    def test_returns_false_when_below_threshold(self):
        lm = DummyLM([{"reasoning": "x", "quality_score": "0.5", "feedback": "Weak."}])
        with dspy.context(lm=lm):
            metric = make_quality_metric()
            example = dspy.Example(description="A chatbot").with_inputs("description")
            pred = dspy.Example(prompt_text="Bad prompt.")
            result = metric(example, pred, trace="some_trace")
        assert result is False  # 0.5 < QUALITY_THRESHOLD (0.7)


class TestMakeComparisonMetric:
    def test_returns_float_score(self):
        lm = DummyLM([{"reasoning": "x", "improvement_score": "0.9", "feedback": "Better."}])
        with dspy.context(lm=lm):
            metric = make_comparison_metric()
            example = dspy.Example(
                current_prompt="Old", change_request="Improve"
            ).with_inputs("current_prompt", "change_request")
            pred = dspy.Example(improved_prompt="New and improved.")
            score = metric(example, pred)
        assert score == 0.9

    def test_returns_bool_when_trace_provided(self):
        lm = DummyLM([{"reasoning": "x", "improvement_score": "0.9", "feedback": "Better."}])
        with dspy.context(lm=lm):
            metric = make_comparison_metric()
            example = dspy.Example(
                current_prompt="Old", change_request="Improve"
            ).with_inputs("current_prompt", "change_request")
            pred = dspy.Example(improved_prompt="New.")
            result = metric(example, pred, trace="t")
        assert result is True


class TestConvenienceMetricFunctions:
    def test_prompt_quality_metric_returns_float(self):
        """Module-level alias returns a float score."""
        lm = DummyLM([
            {"reasoning": "Good", "quality_score": "0.8", "feedback": "Nice"},
            {"reasoning": "Good", "quality_score": "0.9", "feedback": "Great"},
        ])
        with dspy.context(lm=lm):
            example = dspy.Example(description="test").with_inputs("description")
            pred = dspy.Example(prompt_text="You are a helpful assistant.")
            score1 = prompt_quality_metric(example, pred)
            score2 = prompt_quality_metric(example, pred)
            assert isinstance(score1, float)
            assert isinstance(score2, float)

    def test_prompt_comparison_metric_returns_float(self):
        """Module-level alias returns a float score."""
        lm = DummyLM([
            {"reasoning": "Good", "improvement_score": "0.7", "feedback": "OK"},
            {"reasoning": "Good", "improvement_score": "0.95", "feedback": "Excellent"},
        ])
        with dspy.context(lm=lm):
            example = dspy.Example(
                current_prompt="Old prompt", change_request="Make it better"
            ).with_inputs("current_prompt", "change_request")
            pred = dspy.Example(improved_prompt="New prompt")
            score1 = prompt_comparison_metric(example, pred)
            score2 = prompt_comparison_metric(example, pred)
            assert isinstance(score1, float)
            assert isinstance(score2, float)

    def test_prompt_quality_metric_returns_bool_with_trace(self):
        """Convenience function respects trace argument like the factory version."""
        lm = DummyLM([
            {"reasoning": "x", "quality_score": "0.85", "feedback": "Good."},
        ])
        with dspy.context(lm=lm):
            example = dspy.Example(description="test").with_inputs("description")
            pred = dspy.Example(prompt_text="You are a helpful assistant.")
            result = prompt_quality_metric(example, pred, trace="some_trace")
            assert result is True  # 0.85 >= QUALITY_THRESHOLD (0.7)

    def test_prompt_comparison_metric_returns_bool_with_trace(self):
        """Convenience function respects trace argument like the factory version."""
        lm = DummyLM([
            {"reasoning": "x", "improvement_score": "0.5", "feedback": "Weak."},
        ])
        with dspy.context(lm=lm):
            example = dspy.Example(
                current_prompt="Old", change_request="Improve"
            ).with_inputs("current_prompt", "change_request")
            pred = dspy.Example(improved_prompt="New")
            result = prompt_comparison_metric(example, pred, trace="t")
            assert result is False  # 0.5 < QUALITY_THRESHOLD (0.7)
