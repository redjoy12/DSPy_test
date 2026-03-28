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
        judge: PromptQualityJudge | None = None,
        store: PromptStore | None = None,
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
        model: str = "openai/gpt-4o-mini",
    ) -> PromptVersion:
        """Generate a prompt from a description, evaluate it, and save a versioned record.

        Args:
            name: Identifier used to group prompt versions in the store.
            description: Human-language description of what the prompt should do.
            context: Optional target-audience, tone, or constraint hints.
            model: Recorded in version metadata for audit/traceability purposes
                only. This value does **not** control which LLM is used for
                generation — the active LLM is determined globally by
                ``configure_lm()`` (see ``src/config.py``).

        Returns:
            The newly created ``PromptVersion``.
        """
        result = self.forward(description=description, context=context)
        score, feedback = self.judge.evaluate_quality(
            prompt_text=result.prompt_text,
            description=description,
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
            model=model,
        )
        self.store.save(name, version)
        return version
