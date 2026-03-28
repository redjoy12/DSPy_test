import dspy

QUALITY_THRESHOLD = 0.7


class PromptQualitySignature(dspy.Signature):
    """Score a prompt's quality on clarity, completeness, specificity, and best practices adherence."""
    prompt_text: str = dspy.InputField(desc="the prompt to evaluate")
    original_description: str = dspy.InputField(desc="what the prompt was supposed to do")
    quality_score: float = dspy.OutputField(desc="quality score from 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="specific suggestions for improvement")


class PromptComparisonSignature(dspy.Signature):
    """Compare an improved prompt against the original and the requested change."""
    original_prompt: str = dspy.InputField(desc="the original prompt before changes")
    improved_prompt: str = dspy.InputField(desc="the modified prompt after changes")
    change_request: str = dspy.InputField(desc="what changes were requested")
    improvement_score: float = dspy.OutputField(desc="how well the changes address the request, 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="assessment of the changes made")


class PromptQualityJudge:
    def __init__(
        self,
        judge_module=None,
        comparison_module=None,
    ):
        self.judge = judge_module or dspy.ChainOfThought(PromptQualitySignature)
        self.comparison = comparison_module or dspy.ChainOfThought(PromptComparisonSignature)

    def evaluate_quality(self, prompt_text: str, description: str) -> tuple[float, str]:
        result = self.judge(prompt_text=prompt_text, original_description=description)
        try:
            score = max(0.0, min(1.0, float(result.quality_score)))
        except (ValueError, TypeError):
            score = 0.0  # Conservative fallback for unparseable scores
        return score, result.feedback

    def evaluate_comparison(
        self,
        original_prompt: str,
        improved_prompt: str,
        change_request: str,
    ) -> tuple[float, str]:
        result = self.comparison(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            change_request=change_request,
        )
        try:
            score = max(0.0, min(1.0, float(result.improvement_score)))
        except (ValueError, TypeError):
            score = 0.0  # Conservative fallback for unparseable scores
        return score, result.feedback


def make_quality_metric(judge=None):
    """Create a DSPy-compatible metric function for prompt quality evaluation."""
    judge = judge or PromptQualityJudge()

    def metric(example, pred, trace=None) -> float:
        score, _ = judge.evaluate_quality(
            prompt_text=pred.prompt_text,
            description=example.description,
        )
        if trace is not None:
            return score >= QUALITY_THRESHOLD
        return score

    return metric


def make_comparison_metric(judge=None):
    """Create a DSPy-compatible metric function for prompt improvement evaluation."""
    judge = judge or PromptQualityJudge()

    def metric(example, pred, trace=None) -> float:
        score, _ = judge.evaluate_comparison(
            original_prompt=example.current_prompt,
            improved_prompt=pred.improved_prompt,
            change_request=example.change_request,
        )
        if trace is not None:
            return score >= QUALITY_THRESHOLD
        return score

    return metric


_cached_quality_metric = None
_cached_comparison_metric = None


def prompt_quality_metric(example, pred, trace=None) -> float:
    """Convenience wrapper that caches the quality metric on first call."""
    global _cached_quality_metric
    if _cached_quality_metric is None:
        _cached_quality_metric = make_quality_metric()
    return _cached_quality_metric(example, pred, trace)


def prompt_comparison_metric(example, pred, trace=None) -> float:
    """Convenience wrapper that caches the comparison metric on first call."""
    global _cached_comparison_metric
    if _cached_comparison_metric is None:
        _cached_comparison_metric = make_comparison_metric()
    return _cached_comparison_metric(example, pred, trace)
