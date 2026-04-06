import dspy


class ExtractPatternSignature(dspy.Signature):
    """Extract the general pattern and root cause from failing examples.

    Transform concrete failing examples into abstract patterns that describe
    the underlying issue without copying specific scenarios verbatim."""

    failing_examples: str = dspy.InputField(
        desc="concrete failing examples (input/output pairs or conversation logs)"
    )
    change_request: str = dspy.InputField(
        desc="what the user wants to add, modify, or fix"
    )
    issue: str = dspy.OutputField(
        desc="general description of what's wrong (e.g., 'Bot responds generically to specific queries')"
    )
    pattern: str = dspy.OutputField(
        desc="what the user was trying to do and what happened (general pattern, not specific case)"
    )
    root_cause: str = dspy.OutputField(
        desc="what the prompt is missing that causes this failure"
    )


class PatternExtractor(dspy.Module):
    def __init__(self, extractor_module=None):
        self.extract = extractor_module or dspy.ChainOfThought(ExtractPatternSignature)

    def forward(
        self,
        failing_examples: str,
        change_request: str = "",
    ) -> tuple[str, str, str]:
        """Extract abstract pattern from failing examples.

        Args:
            failing_examples: Concrete failing examples (string or structured)
            change_request: Optional context about what user wants to fix

        Returns:
            Tuple of (issue, pattern, root_cause) as strings

        Raises:
            RuntimeError: If LLM fails or returns invalid output
        """
        try:
            result = self.extract(
                failing_examples=failing_examples,
                change_request=change_request,
            )
        except Exception as e:
            raise RuntimeError(f"Pattern extraction failed: {e}") from e

        issue = result.issue.strip() if result.issue else ""
        pattern = result.pattern.strip() if result.pattern else ""
        root_cause = result.root_cause.strip() if result.root_cause else ""

        if not issue and not pattern and not root_cause:
            raise RuntimeError("Pattern extraction returned empty results")

        return issue, pattern, root_cause


def format_abstracted_patterns(issue: str, pattern: str, root_cause: str) -> str:
    """Format extracted patterns into a readable string for the LLM.

    Args:
        issue: The extracted issue description
        pattern: The extracted pattern description
        root_cause: The extracted root cause

    Returns:
        Formatted string with all non-empty components

    Raises:
        ValueError: If all components are empty
    """
    if not issue and not pattern and not root_cause:
        raise ValueError("Cannot format empty abstracted patterns")

    parts = []
    if issue:
        parts.append(f"Issue: {issue}")
    if pattern:
        parts.append(f"Pattern: {pattern}")
    if root_cause:
        parts.append(f"Root cause: {root_cause}")

    return "\n".join(parts)
