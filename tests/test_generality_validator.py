import pytest
from src.validation.generality_validator import (
    check_literal_copy,
    validate_generalization,
    ValidationResult,
)


class TestCheckLiteralCopy:
    def test_no_examples_returns_valid(self):
        result = check_literal_copy("Some prompt", "")
        assert result.is_valid is True

    def test_no_prompt_returns_valid(self):
        result = check_literal_copy("", "Input: foo -> Output: bar")
        assert result.is_valid is True

    def test_detects_exact_match(self):
        failing = "Input: 'How much will my refund be?' -> Output: 'Your refund will be processed soon.'"
        prompt = "When user asks 'How much will my refund be?', respond with 'Your refund will be processed soon'."
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is False
        assert len(result.detected_literals) > 0

    def test_no_literal_copy(self):
        failing = "Input: 'How much will my refund be?' -> Output: 'Your refund will be processed soon.'"
        prompt = "When users ask about refund amounts, provide specific calculated values with details."
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True

    def test_detects_fuzzy_match(self):
        failing = "Input: 'What is my order status?' -> Output: 'I don't know'"
        prompt = "When user asks what is my order status, say I don't know please check the website"
        result = check_literal_copy(prompt, failing, match_threshold=0.7)
        assert result.is_valid is False

    def test_handles_unicode_without_error(self):
        failing = "Input: '你的订单在哪里？' -> Output: '我找不到你的订单'"
        prompt = "当用户问你的订单在哪里时，回复说我找不到你的订单"
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True

    def test_handles_cjk_characters_in_quotes(self):
        failing = "Input: '你好世界' -> Output: '再见世界'"
        prompt = "Handle input 你好世界 with output 再见世界"
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True

    def test_no_false_positives_on_unicode(self):
        failing = "Input: 'Question?' -> Output: 'Answer.'"
        prompt = "Be helpful and answer questions about your status."
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True

    def test_handles_cjk_characters_in_quotes(self):
        failing = "Input: '你好世界' -> Output: '再见世界'"
        prompt = "Handle input 你好世界 with output 再见世界"
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True

    def test_no_false_positives_on_unicode(self):
        failing = "Input: 'Question?' -> Output: 'Answer.'"
        prompt = "Be helpful and answer questions about your status."
        result = check_literal_copy(prompt, failing)
        assert result.is_valid is True


class TestValidateGeneralization:
    def test_valid_prompt_with_no_examples(self):
        result = validate_generalization("Some improved prompt", "")
        assert result.is_valid is True

    def test_empty_improved_prompt(self):
        result = validate_generalization("", "Input: foo -> bar")
        assert result.is_valid is False

    def test_too_short_improved_prompt(self):
        result = validate_generalization("Hi", "Input: foo -> bar")
        assert result.is_valid is False
