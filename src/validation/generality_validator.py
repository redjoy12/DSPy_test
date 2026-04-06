import re
from dataclasses import dataclass


STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "also",
        "as",
        "each",
        "every",
        "few",
        "more",
        "most",
        "nor",
        "other",
        "own",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "here",
        "there",
        "then",
        "about",
        "into",
        "up",
        "down",
        "if",
        "while",
        "return",
    }
)


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    detected_literals: list[str] | None = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and removing extra whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_key_phrases(text: str) -> list[str]:
    """Extract potential key phrases from text that might be copied verbatim."""
    phrases = []

    input_pattern = re.findall(r'input:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    phrases.extend(input_pattern)

    output_pattern = re.findall(r'output:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    phrases.extend(output_pattern)

    expected_pattern = re.findall(
        r'expected:\s*["\']([^"\']+)["\']', text, re.IGNORECASE
    )
    phrases.extend(expected_pattern)

    actual_pattern = re.findall(r'actual:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
    phrases.extend(actual_pattern)

    for match in re.finditer(r"(?:human|user|assistant):\s*(.+)", text, re.IGNORECASE):
        if len(match.group(1).strip()) > 10:
            phrases.append(match.group(1).strip()[:100])

    return [p for p in phrases if len(p) > 3]


def find_consecutive_matches(phrase: str, text: str, min_consecutive: int = 3) -> bool:
    """Check if phrase has at least min_consecutive consecutive words in text."""
    words = phrase.lower().split()
    if len(words) < min_consecutive:
        return False

    text_lower = text.lower()
    for i in range(len(words) - min_consecutive + 1):
        subsequence = " ".join(words[i : i + min_consecutive])
        if subsequence in text_lower:
            return True
    return False


def check_literal_copy(
    improved_prompt: str,
    failing_examples: str,
    min_phrase_length: int = 15,
    match_threshold: float = 0.8,
    min_consecutive: int = 3,
) -> ValidationResult:
    """Check if improved prompt contains literal content from failing examples.

    Args:
        improved_prompt: The generated improved prompt
        failing_examples: Original failing examples provided by user
        min_phrase_length: Minimum length for phrase matching
        match_threshold: Similarity threshold (0-1) for fuzzy matching
        min_consecutive: Minimum consecutive words to flag as copy

    Returns:
        ValidationResult with is_valid, reason, and any detected literals
    """
    if not failing_examples or not improved_prompt:
        return ValidationResult(is_valid=True, reason="No examples to check")

    key_phrases = extract_key_phrases(failing_examples)
    detected_literals = []

    improved_normalized = normalize_text(improved_prompt)

    for phrase in key_phrases:
        if len(phrase) < min_phrase_length:
            continue

        phrase_lower = phrase.lower()

        exact_pattern = r"\b" + re.escape(phrase_lower) + r"\b"
        if re.search(exact_pattern, improved_normalized):
            detected_literals.append(
                phrase[:50] + "..." if len(phrase) > 50 else phrase
            )
            continue

        if find_consecutive_matches(phrase, improved_normalized, min_consecutive):
            detected_literals.append(
                phrase[:50] + "..." if len(phrase) > 50 else phrase
            )
            continue

        words = phrase_lower.split()
        if len(words) >= 4:
            non_stop_words = [w for w in words if w not in STOP_WORDS]
            if not non_stop_words:
                continue
            matching_non_stop = sum(
                1 for w in non_stop_words if w in improved_normalized
            )
            similarity = matching_non_stop / len(non_stop_words)
            if similarity >= match_threshold:
                detected_literals.append(
                    phrase[:50] + "..." if len(phrase) > 50 else phrase
                )

    if detected_literals:
        return ValidationResult(
            is_valid=False,
            reason=f"Detected {len(detected_literals)} literal phrase(s) from failing examples in improved prompt",
            detected_literals=detected_literals[:5],
        )

    return ValidationResult(
        is_valid=True,
        reason="No literal content detected from failing examples",
    )


def validate_generalization(
    improved_prompt: str,
    failing_examples: str,
) -> ValidationResult:
    """Validate that the improved prompt is general and doesn't copy specific scenarios.

    Args:
        improved_prompt: The generated improved prompt
        failing_examples: The input that was passed to the LLM (can be abstracted or original)

    Returns:
        ValidationResult with validation outcome
    """
    literal_check = check_literal_copy(improved_prompt, failing_examples)

    if not literal_check.is_valid:
        return literal_check

    if not improved_prompt or len(improved_prompt.strip()) < 10:
        return ValidationResult(
            is_valid=False,
            reason="Improved prompt is too short or empty",
        )

    return ValidationResult(
        is_valid=True,
        reason="Validation passed - no literal content detected",
    )
