import dspy

from src.evaluation.judge import PromptQualityJudge
from src.store.prompt_store import PromptStore, PromptVersion


class IteratePromptSignature(dspy.Signature):
    """Improve an existing prompt based on a change request and optional failing examples.
    Preserve what works well in the original prompt while addressing the requested changes."""
    current_prompt: str = dspy.InputField(desc="the existing prompt to improve")
    change_request: str = dspy.InputField(desc="what to add, modify, or fix")
    failing_examples: str = dspy.InputField(
        desc="optional: input/output pairs where the current prompt fails"
    )
    improved_prompt: str = dspy.OutputField(desc="the updated prompt incorporating the changes")
    changes_made: str = dspy.OutputField(desc="summary of what was changed and why")


class IteratePromptPipeline(dspy.Module):
    def __init__(
        self,
        generate_module=None,
        judge: PromptQualityJudge | None = None,
        store: PromptStore | None = None,
    ):
        super().__init__()
        self.generate = generate_module or dspy.ChainOfThought(IteratePromptSignature)
        self.judge = judge or PromptQualityJudge()
        self.store = store or PromptStore()

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
        current_prompt: str | None = None,
        description: str | None = None,
        failing_examples: str = "",
        model: str | None = None,
        min_score: float | None = None,
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
                prompt fails, used to guide improvements.
            model: The LLM to use for generation and judging.  When provided,
                a ``dspy.LM`` is created and used for this call.  When
                ``None`` (the default), the globally configured LM is used.
            min_score: Optional minimum quality score threshold. If provided and
                the iterated prompt scores below this value, a ``ValueError``
                is raised and the prompt is **not** saved.

        Returns:
            The newly created ``PromptVersion``.

        Raises:
            ValueError: If *min_score* is set and the prompt scores below it.
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

        if model is not None:
            lm = dspy.LM(model)
            ctx = dspy.context(lm=lm)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            result = self.forward(
                current_prompt=current_prompt,
                change_request=change_request,
                failing_examples=failing_examples,
            )

            score, feedback = self.judge.evaluate_comparison(
                original_prompt=current_prompt,
                improved_prompt=result.improved_prompt,
                change_request=change_request,
            )

        if min_score is not None and score < min_score:
            raise ValueError(
                f"Generated prompt scored {score:.2f}, below minimum threshold {min_score:.2f}. "
                f"Feedback: {feedback}"
            )

        # Record the actual model used in metadata.
        actual_model = model or getattr(getattr(dspy.settings, 'lm', None), 'model', 'unknown')
        version_num = self.store.get_next_version(name)
        version = PromptVersion(
            version=version_num,
            parent_version=parent_version,
            prompt_text=result.improved_prompt,
            description=description,
            change_request=change_request,
            changes_made=result.changes_made,
            quality_score=score,
            judge_feedback=feedback,
            pipeline="iterate",
            model=actual_model,
        )
        self.store.save(name, version)
        return version
