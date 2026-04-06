import pytest
import dspy
from dspy.utils import DummyLM

from src.evaluation.example_metric import ExampleBasedMetric


class TestExampleBasedMetric:
    def test_perfect_score_when_all_examples_match(self):
        lm = DummyLM([
            {"output": "Hello! How can I help you?"},
            {"output": "Sure, I can help with that."},
        ])
        examples = [
            {"input": "Hi", "expected_output": "Hello! How can I help you?"},
            {"input": "Can you help?", "expected_output": "Sure, I can help with that."},
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("You are helpful.", examples)
        assert score == 1.0

    def test_partial_score_when_some_match(self):
        lm = DummyLM([
            {"output": "Hello! How can I help you?"},
            {"output": "Wrong output entirely."},
        ])
        examples = [
            {"input": "Hi", "expected_output": "Hello! How can I help you?"},
            {"input": "Can you help?", "expected_output": "Sure, I can help with that."},
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("You are helpful.", examples)
        assert score == 0.5

    def test_zero_score_when_nothing_matches(self):
        lm = DummyLM([{"output": "Completely wrong."}])
        examples = [{"input": "Hi", "expected_output": "Hello"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Bad prompt", examples)
        assert score == 0.0

    def test_empty_examples_returns_zero(self):
        """No examples means nothing to score — returns 0.0 without calling LM."""
        metric = ExampleBasedMetric()
        score = metric.evaluate("Some prompt", [])
        assert score == 0.0

    def test_case_insensitive_matching(self):
        """The metric compares outputs case-insensitively."""
        lm = DummyLM([{"output": "HELLO WORLD"}])
        examples = [{"input": "greet", "expected_output": "hello world"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Greeter", examples)
        assert score == 1.0

    def test_whitespace_trimming(self):
        """Leading/trailing whitespace is stripped before comparison."""
        lm = DummyLM([{"output": "  Hello  "}])
        examples = [{"input": "greet", "expected_output": "Hello"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Greeter", examples)
        assert score == 1.0

    def test_partial_token_overlap_gives_fractional_score(self):
        """When the output partially overlaps with the expected, a fractional score is returned."""
        # actual: "the cat sat on the mat" -> tokens: {the, cat, sat, on, mat}
        # expected: "the dog sat on the rug" -> tokens: {the, dog, sat, on, rug}
        # shared: {the, sat, on} = 3,  union: {the, cat, sat, on, mat, dog, rug} = 7
        # Jaccard = 3/7
        lm = DummyLM([{"output": "the cat sat on the mat"}])
        examples = [{"input": "story", "expected_output": "the dog sat on the rug"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Storyteller", examples)
        assert score == pytest.approx(3 / 7)

    def test_no_token_overlap_gives_zero(self):
        """When actual and expected share no tokens, the score is 0.0."""
        lm = DummyLM([{"output": "apples oranges bananas"}])
        examples = [{"input": "colors", "expected_output": "red green blue"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Colors", examples)
        assert score == 0.0

    def test_complete_token_overlap_but_different_text(self):
        """When all expected tokens are present but output has extra tokens,
        the score is less than 1.0 (Jaccard penalises extra tokens)."""
        # actual: "hello world how are you" -> tokens: {hello, world, how, are, you} = 5
        # expected: "hello world" -> tokens: {hello, world} = 2
        # shared: {hello, world} = 2,  union = 5
        # Jaccard = 2/5
        lm = DummyLM([{"output": "hello world how are you"}])
        examples = [{"input": "greet", "expected_output": "hello world"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Greeter", examples)
        assert score == pytest.approx(2 / 5)

    def test_mixed_exact_and_partial_match(self):
        """One exact match (1.0) plus one partial match averages correctly."""
        # First example: exact match -> 1.0
        # Second example: actual "I can help" vs expected "I can assist"
        #   tokens actual: {i, can, help}, expected: {i, can, assist}
        #   shared: {i, can} = 2,  union: {i, can, help, assist} = 4
        #   Jaccard = 2/4 = 0.5
        # Average: (1.0 + 0.5) / 2 = 0.75
        lm = DummyLM([
            {"output": "Hello!"},
            {"output": "I can help"},
        ])
        examples = [
            {"input": "Hi", "expected_output": "Hello!"},
            {"input": "Need assistance", "expected_output": "I can assist"},
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Helper", examples)
        assert score == pytest.approx(0.75)

    def test_punctuation_invariant_matching(self):
        lm = DummyLM([{"output": "Hello, world!"}])
        examples = [{"input": "greet", "expected_output": "Hello world"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate("Greeter", examples)
        assert score == 1.0

    def test_all_predictions_fail_raises_runtime_error(self):
        lm = DummyLM([])  # no responses → will raise
        examples = [{"input": "test", "expected_output": "output"}]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            with pytest.raises(RuntimeError, match="All 1 example evaluations failed"):
                metric.evaluate("Test prompt", examples)


class TestExampleBasedMetricMultiTurn:
    def test_empty_structured_examples_returns_zero(self):
        metric = ExampleBasedMetric()
        assert metric.evaluate_multi_turn("Some prompt", []) == 0.0

    def test_improved_prompt_avoiding_bad_output_scores_high(self):
        """If the new output doesn't match the unsatisfactory output at all,
        the inverted overlap score is 1.0."""
        lm = DummyLM([{"output": "Here is a helpful reply"}])
        examples = [
            {
                "messages": [
                    {"role": "human", "content": "What products?"},
                    {"role": "assistant", "content": "X, Y, Z"},
                    {"role": "human", "content": "Tell me about X"},
                ],
                "unsatisfactory_output": "apples oranges bananas",
            }
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate_multi_turn("Better prompt", examples)
        assert score == 1.0

    def test_improved_prompt_reproducing_bad_output_scores_zero(self):
        """If the new output exactly matches the bad output, score is 0.0."""
        lm = DummyLM([{"output": "I don't know"}])
        examples = [
            {
                "messages": [{"role": "human", "content": "Tell me about X"}],
                "unsatisfactory_output": "I don't know",
            }
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            score = metric.evaluate_multi_turn("Bad prompt", examples)
        assert score == 0.0

    def test_split_conversation_last_human_becomes_current_input(self):
        messages = [
            {"role": "human", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "human", "content": "How are you?"},
        ]
        history, current = ExampleBasedMetric._split_conversation(messages)
        assert current == "How are you?"
        assert "Human: Hi" in history
        assert "Assistant: Hello" in history
        assert "How are you?" not in history

    def test_split_conversation_without_trailing_human(self):
        messages = [
            {"role": "human", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        history, current = ExampleBasedMetric._split_conversation(messages)
        assert current == ""
        assert "Human: Hi" in history
        assert "Assistant: Hello" in history

    def test_all_multi_turn_predictions_fail_raises_runtime_error(self):
        lm = DummyLM([])
        examples = [
            {
                "messages": [{"role": "human", "content": "hi"}],
                "unsatisfactory_output": "bad",
            }
        ]
        with dspy.context(lm=lm):
            metric = ExampleBasedMetric()
            with pytest.raises(RuntimeError, match="All 1 multi-turn evaluations failed"):
                metric.evaluate_multi_turn("p", examples)
