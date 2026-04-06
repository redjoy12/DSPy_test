from __future__ import annotations
import logging
from contextlib import nullcontext
from typing import Callable, Optional

import dspy

from src.evaluation.judge import PromptQualityJudge
from src.pipelines.abstract_pattern import PatternExtractor, format_abstracted_patterns
from src.pipelines.consolidate_prompt import PromptConsolidator
from src.store.prompt_store import PromptStore, PromptVersion
from src.validation.generality_validator import (
    validate_generalization,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class IteratePromptSignature(dspy.Signature):
    """Improve an existing prompt based on a change request and optional failing examples.
    Preserve what works well in the original prompt while addressing the requested changes.

    IMPORTANT: When failing examples are provided, focus on fixing the GENERAL PRINCIPLE
    or ROOT CAUSE of the failure, NOT on adding the specific scenario verbatim.
    The improved prompt should work for any case matching the pattern, not just the
    exact examples given.

    DEDUPLICATION: Before adding a new rule, check whether an existing rule in
    `current_prompt` already expresses the same intent. If so, MERGE the new
    requirement into the existing rule rather than appending a new one. The
    output should not contain two rules that say the same thing in different
    words.

    GENERALITY: Express new rules as general principles. Do NOT name specific
    entities, products, brands, people, numbers, dates, IDs, or quote
    conversation snippets from `failing_examples`. State the rule at the level
    of *categories of input*, not specific inputs.

    Concrete contrast:
      Bad:  "When the user asks about Acme Pro pricing, respond with the
             current monthly cost."
      Good: "When asked about product pricing, look up the current pricing
             source before answering."
    """

    current_prompt: str = dspy.InputField(desc="the existing prompt to improve")
    change_request: str = dspy.InputField(desc="what to add, modify, or fix")
    failing_examples: str = dspy.InputField(
        desc="optional: abstracted patterns describing failures, NOT literal examples to embed"
    )
    improved_prompt: str = dspy.OutputField(
        desc="the updated prompt incorporating the changes as general rules, with overlapping rules merged rather than appended"
    )
    changes_made: str = dspy.OutputField(desc="summary of what was changed and why")


class IteratePromptPipeline(dspy.Module):
    def __init__(
        self,
        generate_module=None,
        judge: Optional[PromptQualityJudge] = None,
        store: Optional[PromptStore] = None,
        pattern_extractor: Optional[PatternExtractor] = None,
        consolidator: Optional[PromptConsolidator] = None,
    ):
        super().__init__()
        self.generate = generate_module or dspy.ChainOfThought(IteratePromptSignature)
        self.judge = judge or PromptQualityJudge()
        self.store = store or PromptStore()
        self.pattern_extractor = pattern_extractor or PatternExtractor()
        self.consolidator = consolidator or PromptConsolidator()

    def forward(
        self,
        current_prompt: str,
        change_request: str,
        failing_examples: str = "",
    ):
        return self.generate(
            current_prompt=current_prompt,
            change_request=change_request,
            failing_examples=failing_examples,
        )

    def iterate_and_save(
        self,
        name: str,
        change_request: str,
        current_prompt: Optional[str] = None,
        description: Optional[str] = None,
        failing_examples: str = "",
        structured_examples: Optional[list[dict]] = None,
        model: Optional[str] = None,
        min_score: Optional[float] = None,
        interactive: bool = False,
        prompt_func: Optional[Callable] = None,
    ) -> PromptVersion:
        """Improve an existing prompt, evaluate the result, and save a new version.

        If ``current_prompt`` is not provided, the latest saved version for
        *name* is loaded automatically.

        Args:
            name: Identifier used to group prompt versions in the store.
            change_request: Description of what to add, modify, or fix.
            current_prompt: The prompt text to iterate on. When ``None``, the
                latest version is loaded from the store.
            description: Optional description carried forward into metadata.
                Defaults to the description from the previous version when
                *current_prompt* is loaded from the store.
            failing_examples: Optional input/output pairs where the current
                prompt fails, used to guide improvements. When
                *structured_examples* is also provided, the serialized
                structured examples are appended to this string so both
                formats can be supplied simultaneously.
            structured_examples: Optional list of multi-turn conversation
                examples that the current system prompt handles poorly. Each
                example is a dict with ``messages`` (list of
                ``{"role": "human"|"assistant", "content": ...}``) and an
                ``unsatisfactory_output`` string. The structured examples are
                serialized into the ``failing_examples`` string input for the
                DSPy signature and stored verbatim on the new version for
                later reference.
            model: The LLM to use for generation and judging.  When provided,
                a ``dspy.LM`` is created and used for this call.  When
                ``None`` (the default), the globally configured LM is used.
            min_score: Optional minimum quality score threshold. If provided and
                the iterated prompt scores below this value, a ``ValueError``
                is raised and the prompt is **not** saved.
            interactive: If ``True``, prompt user when validation fails for
                (a)ccept/(r)etry/(e)xit choice. If ``False`` (default), raise
                ``ValueError`` immediately on validation failure.
            prompt_func: Optional callable for user prompts in interactive mode.
                Defaults to built-in ``input()``. Useful for testing or
                non-interactive environments.

        Returns:
            The newly created ``PromptVersion``.

        Raises:
            ValueError: If *min_score* is set and the prompt scores below it,
                or if validation fails and the user rejects the prompt.
        """
        if current_prompt is None:
            latest = self.store.load_latest(name)
            current_prompt = latest.prompt_text
            parent_version = latest.version
            description = description or latest.description
        else:
            existing = self.store.list_versions(name)
            parent_version = max(existing) if existing else None
            description = description or ""

        original_failing_examples = failing_examples

        if structured_examples:
            structured_text = PromptVersion.format_examples_as_text(structured_examples)
            if failing_examples and structured_text:
                failing_examples = f"{failing_examples}\n{structured_text}"
            else:
                failing_examples = structured_text or failing_examples

        if model is not None:
            lm = dspy.LM(model)
            ctx = dspy.context(lm=lm)
        else:
            ctx = nullcontext()

        try:
            with ctx:
                abstracted_pattern = ""
                pattern_extraction_failed = False
                if failing_examples:
                    try:
                        issue, pattern, root_cause = self.pattern_extractor(
                            failing_examples=failing_examples,
                            change_request=change_request,
                        )
                        abstracted_pattern = format_abstracted_patterns(
                            issue, pattern, root_cause
                        )
                    except Exception as e:
                        logger.warning(
                            f"Pattern extraction failed, falling back to raw examples: {e}"
                        )
                        pattern_extraction_failed = True

                iteration_input = (
                    abstracted_pattern if abstracted_pattern else failing_examples
                )
                result = self(
                    current_prompt=current_prompt,
                    change_request=change_request,
                    failing_examples=iteration_input,
                )

                # Post-pass: scrub scenario specifics + merge redundant rules.
                # The consolidated prompt is what gets validated, judged, and
                # saved. On failure we fall back to the raw improved_prompt
                # rather than killing the iteration.
                try:
                    consolidated_prompt, consolidation_notes = self.consolidator(
                        raw_prompt=result.improved_prompt,
                        original_prompt=current_prompt,
                        change_request=change_request,
                        abstracted_pattern=abstracted_pattern,
                    )
                except Exception as e:
                    logger.warning(
                        f"Consolidation failed, using raw improved prompt: {e}"
                    )
                    consolidated_prompt = result.improved_prompt
                    consolidation_notes = ""

                validation = validate_generalization(
                    improved_prompt=consolidated_prompt,
                    failing_examples=iteration_input,
                )

                if original_failing_examples and abstracted_pattern:
                    original_validation = validate_generalization(
                        improved_prompt=consolidated_prompt,
                        failing_examples=original_failing_examples,
                    )
                    if not original_validation.is_valid:
                        validation = original_validation
                    elif not validation.is_valid:
                        pass

                validation_passed = validation.is_valid

                if not validation_passed:
                    if interactive:
                        prompt_fn = prompt_func or input
                        while True:
                            response = (
                                prompt_fn(
                                    f"Validation failed: {validation.reason}\n"
                                    f"Detected literal content: {validation.detected_literals or []}\n"
                                    f"Improved prompt:\n{consolidated_prompt}\n\n"
                                    "Do you want to (a)ccept/(r)etry/(e)xit? "
                                )
                                .strip()
                                .lower()
                            )
                            if response in ("a", "accept"):
                                validation_passed = True
                                break
                            elif response in ("r", "retry"):
                                raise ValueError(
                                    "User requested retry after validation failure"
                                )
                            elif response in ("e", "exit"):
                                raise ValueError("Aborted by user")
                    else:
                        raise ValueError(
                            f"Validation failed: {validation.reason}\n"
                            f"Detected literal content: {validation.detected_literals or []}\n"
                            f"Improved prompt:\n{consolidated_prompt}"
                        )

                score, feedback = self.judge.evaluate_comparison(
                    original_prompt=current_prompt,
                    improved_prompt=consolidated_prompt,
                    change_request=change_request,
                )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Prompt iteration failed: {e}") from e

        if min_score is not None and score < min_score:
            raise ValueError(
                f"Generated prompt scored {score:.2f}, below minimum threshold {min_score:.2f}. "
                f"Feedback: {feedback}"
            )

        enhanced_changes = self._build_changes_made(
            original_examples=original_failing_examples,
            structured_examples=structured_examples,
            abstracted_pattern=abstracted_pattern,
            result_changes=result.changes_made,
            validation_result=validation,
            consolidation_notes=consolidation_notes,
        )

        actual_model = model or self._detect_model()
        version = PromptVersion(
            version=None,
            parent_version=parent_version,
            prompt_text=consolidated_prompt,
            description=description,
            change_request=change_request,
            changes_made=enhanced_changes,
            structured_examples=structured_examples,
            abstracted_patterns=abstracted_pattern,
            validation_passed=validation_passed,
            quality_score=score,
            judge_feedback=feedback,
            pipeline="iterate",
            model=actual_model,
        )
        version_num, _ = self.store.get_and_save_version(name, version)
        version.version = version_num
        return version

    def _build_changes_made(
        self,
        original_examples: str,
        structured_examples: Optional[list[dict]],
        abstracted_pattern: str,
        result_changes: str,
        validation_result: ValidationResult,
        consolidation_notes: str = "",
    ) -> str:
        """Build enhanced changes_made string with transparency info."""
        lines = []

        if original_examples or structured_examples:
            lines.append("--- Original Failing Examples ---")
            if original_examples:
                lines.append(original_examples)
            if structured_examples:
                lines.append(PromptVersion.format_examples_as_text(structured_examples))
            lines.append("")

        if abstracted_pattern:
            lines.append("--- Abstracted Pattern (derived from examples) ---")
            lines.append(abstracted_pattern)
            lines.append("")

        lines.append(f"--- Changes Made ---\n{result_changes}")
        lines.append("")

        if consolidation_notes:
            lines.append(f"--- Consolidation (scrub + merge) ---\n{consolidation_notes}")
            lines.append("")

        lines.append(f"--- Validation ---\n{validation_result.reason}")

        return "\n".join(lines)

    def _detect_model(self) -> str:
        """Detect the currently configured model from DSPy settings."""
        lm = getattr(dspy.settings, "lm", None)
        if lm is None:
            return "unknown"
        model = getattr(lm, "model", None)
        return model if model else "unknown"
