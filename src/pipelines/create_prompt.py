from __future__ import annotations
from typing import Optional

import dspy

from src.evaluation.judge import PromptQualityJudge
from src.store.prompt_store import PromptStore, PromptVersion


class CreatePromptSignature(dspy.Signature):
    """Generate a high-quality LLM prompt from a human language description.
    The prompt should include a clear role, specific instructions, constraints,
    and output format guidance as appropriate."""

    description: str = dspy.InputField(desc="what the prompt should do")
    context: str = dspy.InputField(desc="optional: target audience, tone, constraints")
    prompt_text: str = dspy.OutputField(desc="the complete, ready-to-use prompt")
    reasoning: str = dspy.OutputField(desc="why this prompt structure was chosen")


class CreatePromptPipeline(dspy.Module):
    def __init__(
        self,
        generate_module=None,
        judge: Optional[PromptQualityJudge] = None,
        store: Optional[PromptStore] = None,
    ):
        super().__init__()
        self.generate = generate_module or dspy.ChainOfThought(CreatePromptSignature)
        self.judge = judge or PromptQualityJudge()
        self.store = store or PromptStore()

    def forward(self, description: str, context: str = ""):
        return self.generate(description=description, context=context)

    def create_and_save(
        self,
        name: str,
        description: str,
        context: str = "",
        model: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> PromptVersion:
        """Generate a prompt from a description, evaluate it, and save a versioned record.

        Args:
            name: Identifier used to group prompt versions in the store.
            description: Human-language description of what the prompt should do.
            context: Optional target-audience, tone, or constraint hints.
            model: The LLM to use for generation and judging.  When provided,
                a ``dspy.LM`` is created and used for this call.  When
                ``None`` (the default), the globally configured LM is used.
            min_score: Optional minimum quality score threshold. If provided and
                the generated prompt scores below this value, a ``ValueError``
                is raised and the prompt is **not** saved.

        Returns:
            The newly created ``PromptVersion``.

        Raises:
            ValueError: If *min_score* is set and the prompt scores below it.
        """
        if model is not None:
            lm = dspy.LM(model)
            ctx = dspy.context(lm=lm)
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            result = self.forward(description=description, context=context)
            score, feedback = self.judge.evaluate_quality(
                prompt_text=result.prompt_text,
                description=description,
            )

        if min_score is not None and score < min_score:
            raise ValueError(
                f"Generated prompt scored {score:.2f}, below minimum threshold {min_score:.2f}. "
                f"Feedback: {feedback}"
            )

        # Record the actual model used in metadata.
        actual_model = model or getattr(
            getattr(dspy.settings, "lm", None), "model", "unknown"
        )
        version_num = self.store.get_next_version(name)
        version = PromptVersion(
            version=version_num,
            parent_version=None,
            prompt_text=result.prompt_text,
            description=description,
            quality_score=score,
            judge_feedback=feedback,
            pipeline="create",
            model=actual_model,
        )
        self.store.save(name, version)
        return version
