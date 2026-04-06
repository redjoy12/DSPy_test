import logging
import re

import dspy

logger = logging.getLogger(__name__)


class TestPromptSignature(dspy.Signature):
    """Execute a prompt against a given input and produce the output."""
    system_prompt: str = dspy.InputField(desc="the system prompt to test")
    user_input: str = dspy.InputField(desc="the user input to process")
    output: str = dspy.OutputField(desc="the response following the system prompt's instructions")


class MultiTurnTestSignature(dspy.Signature):
    """Execute a system prompt against a multi-turn conversation and produce
    the assistant's next response."""
    system_prompt: str = dspy.InputField(desc="the system prompt to test")
    conversation_history: str = dspy.InputField(
        desc="previous conversation turns formatted as 'Human: ...' / 'Assistant: ...' lines"
    )
    current_input: str = dspy.InputField(desc="the final human message awaiting a response")
    output: str = dspy.OutputField(
        desc="the assistant's reply following the system prompt's instructions"
    )


class ExampleBasedMetric:
    def __init__(self, predict_module=None, multi_turn_predict_module=None):
        self.predict = predict_module or dspy.Predict(TestPromptSignature)
        self.multi_turn_predict = multi_turn_predict_module or dspy.Predict(
            MultiTurnTestSignature
        )

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

    @staticmethod
    def _split_conversation(messages: list[dict]) -> tuple[str, str]:
        """Split a list of conversation messages into (history_text, current_input).

        The *current_input* is the final human message when (and only when)
        the conversation ends with a human turn; everything before it becomes
        the formatted history. If the conversation ends with an assistant
        turn (or anything other than a human turn), the full list is treated
        as history and ``current_input`` is empty.
        """
        if not messages:
            return "", ""

        last_role = str(messages[-1].get("role", "")).strip().lower()
        if last_role == "human":
            history_msgs = messages[:-1]
            current_input = str(messages[-1].get("content", "")).strip()
        else:
            history_msgs = messages
            current_input = ""

        history_lines: list[str] = []
        for msg in history_msgs:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role == "human":
                label = "Human"
            elif role == "assistant":
                label = "Assistant"
            else:
                label = role.capitalize() or "Unknown"
            history_lines.append(f"{label}: {content}")
        return "\n".join(history_lines), current_input

    def evaluate_multi_turn(
        self, prompt_text: str, structured_examples: list[dict]
    ) -> float:
        """Score a system prompt against a set of multi-turn failing examples.

        For each example, the stored ``messages`` are replayed against
        *prompt_text*: everything up to the last human turn becomes
        ``conversation_history``, the last human turn becomes ``current_input``,
        and the model's output is compared against ``unsatisfactory_output``.

        **Scoring is inverted**: these are *failing* examples, so a good
        improved prompt should produce outputs that differ from the
        unsatisfactory ones. The returned score is ``1 - mean_overlap``
        across examples (1.0 means the new prompt always avoided the bad
        output; 0.0 means it reproduced it exactly).
        """
        if not structured_examples:
            return 0.0

        total_overlap = 0.0
        failures = 0
        for ex in structured_examples:
            try:
                history_text, current_input = self._split_conversation(
                    ex.get("messages", []) or []
                )
                result = self.multi_turn_predict(
                    system_prompt=prompt_text,
                    conversation_history=history_text,
                    current_input=current_input,
                )
                actual = result.output.strip().lower()
                bad = str(ex.get("unsatisfactory_output", "")).strip().lower()

                if actual == bad:
                    overlap = 1.0
                else:
                    overlap = self._token_overlap_score(actual, bad)
                total_overlap += overlap
            except Exception:
                failures += 1
                logger.warning(
                    "Multi-turn prediction failed for example", exc_info=True
                )

        if failures == len(structured_examples):
            raise RuntimeError(
                f"All {len(structured_examples)} multi-turn evaluations failed. "
                "Check LLM configuration and API connectivity."
            )

        if failures > 0:
            logger.warning(
                "%d of %d multi-turn evaluations failed; score based on %d successful evaluations",
                failures, len(structured_examples), len(structured_examples) - failures,
            )

        mean_overlap = total_overlap / len(structured_examples)
        # Invert: high overlap with the bad output = low score.
        return max(0.0, 1.0 - mean_overlap)
