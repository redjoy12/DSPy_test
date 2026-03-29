import logging
import re

import dspy

logger = logging.getLogger(__name__)


class TestPromptSignature(dspy.Signature):
    """Execute a prompt against a given input and produce the output."""
    system_prompt: str = dspy.InputField(desc="the system prompt to test")
    user_input: str = dspy.InputField(desc="the user input to process")
    output: str = dspy.OutputField(desc="the response following the system prompt's instructions")


class ExampleBasedMetric:
    def __init__(self, predict_module=None):
        self.predict = predict_module or dspy.Predict(TestPromptSignature)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip, remove punctuation, and split text into tokens on whitespace."""
        cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
        return cleaned.split()

    @staticmethod
    def _token_overlap_score(actual: str, expected: str) -> float:
        """Compute a token-overlap similarity between *actual* and *expected*.

        Score = |shared_tokens| / |union_of_tokens|  (Jaccard index on tokens).

        Returns 0.0 when both strings are empty or share no tokens.
        """
        actual_tokens = set(ExampleBasedMetric._tokenize(actual))
        expected_tokens = set(ExampleBasedMetric._tokenize(expected))

        if not actual_tokens and not expected_tokens:
            return 0.0

        shared = actual_tokens & expected_tokens
        union = actual_tokens | expected_tokens
        return len(shared) / len(union)

    def evaluate(self, prompt_text: str, examples: list[dict]) -> float:
        if not examples:
            return 0.0

        total_score = 0.0
        failures = 0
        for ex in examples:
            try:
                result = self.predict(
                    system_prompt=prompt_text,
                    user_input=ex["input"],
                )
                actual = result.output.strip().lower()
                expected = ex["expected_output"].strip().lower()

                if actual == expected:
                    total_score += 1.0
                else:
                    total_score += self._token_overlap_score(actual, expected)
            except Exception:
                failures += 1
                logger.warning("Prediction failed for input: %s", ex.get("input", "?"), exc_info=True)

        if failures == len(examples):
            raise RuntimeError(
                f"All {len(examples)} example evaluations failed. "
                "Check LLM configuration and API connectivity."
            )

        if failures > 0:
            logger.warning(
                "%d of %d evaluations failed; score based on %d successful evaluations",
                failures, len(examples), len(examples) - failures,
            )

        return total_score / len(examples)
