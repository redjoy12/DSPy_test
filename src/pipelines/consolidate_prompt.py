import dspy


class ConsolidatePromptSignature(dspy.Signature):
    """Clean and consolidate an iterated prompt before it is saved.

    The input prompt was just produced by another LLM iteration step. It may
    contain two kinds of problems:

    1. Scenario-specific phrasing that was leaked from the failing examples
       used to drive the iteration. Strip any sentence that:
         - names a specific entity, product, brand, person, account, or place
         - cites specific numbers, dates, IDs, prices, or version strings
         - quotes a user/assistant utterance verbatim from a conversation log
         - uses anchors like "for example: ...", "e.g. when the user asks ...",
           "if the customer says ...", "such as the case where ..." that pin a
           rule to one specific scenario instead of stating it as a principle
       Replace such sentences with the equivalent general principle, or drop
       them entirely if a more general rule already covers them.

    2. Redundant rules. Identify rules whose intent overlaps, even when they
       are worded differently, and MERGE them into a single consolidated rule
       that preserves every distinct nuance from the originals. No information
       loss: if two rules disagree or add different qualifications, the merged
       rule must keep both qualifications. Example:
         "Be concise." + "Keep responses short and to the point."
         -> "Be concise: keep responses short and to the point."

    Other rules:
      - Preserve the original prompt's overall structure (role line, sections,
        tone, ordering). Do not aggressively re-format or rewrite the voice.
      - If the input prompt is already general and non-redundant, return it
        essentially unchanged.
      - Never drop a rule whose intent is not already covered elsewhere in the
        prompt.

    Concrete contrast for scrubbing:
      Bad:  "When the user asks about Acme Pro pricing, respond with the
             current monthly cost from the pricing page."
      Good: "When asked about product pricing, look up the current pricing
             source before answering."
    """

    raw_prompt: str = dspy.InputField(
        desc="prompt produced by the iterate step, may contain scenario-specific phrasing or redundant rules"
    )
    change_request: str = dspy.InputField(
        desc="the user's original change request, for context on what the new rules were trying to address"
    )
    abstracted_pattern: str = dspy.InputField(
        desc="optional: the abstracted failure pattern that triggered this iteration; use it to know which themes are legitimate vs. which are scenario detail"
    )
    consolidated_prompt: str = dspy.OutputField(
        desc="the cleaned prompt with scenario-specific content removed and overlapping rules merged"
    )
    consolidation_notes: str = dspy.OutputField(
        desc="brief summary of what was scrubbed and which rules were merged"
    )


class PromptConsolidator(dspy.Module):
    def __init__(self, consolidator_module=None):
        super().__init__()
        self.consolidate = consolidator_module or dspy.ChainOfThought(
            ConsolidatePromptSignature
        )

    def forward(
        self,
        raw_prompt: str,
        change_request: str = "",
        abstracted_pattern: str = "",
    ) -> tuple[str, str]:
        """Run the consolidation pass.

        Args:
            raw_prompt: The improved prompt produced by the iterate step.
            change_request: The user's original change request, for context.
            abstracted_pattern: Optional abstracted failure pattern that drove
                this iteration, used to distinguish legitimate themes from
                scenario detail that should be scrubbed.

        Returns:
            Tuple of (consolidated_prompt, consolidation_notes).

        Raises:
            RuntimeError: If the LLM fails or returns an empty consolidated
                prompt. The caller should fall back to the raw prompt rather
                than letting an empty value reach storage.
        """
        try:
            result = self.consolidate(
                raw_prompt=raw_prompt,
                change_request=change_request,
                abstracted_pattern=abstracted_pattern,
            )
        except Exception as e:
            raise RuntimeError(f"Consolidation failed: {e}") from e

        consolidated = (result.consolidated_prompt or "").strip()
        notes = (result.consolidation_notes or "").strip()

        if not consolidated:
            raise RuntimeError("Consolidation returned an empty prompt")

        return consolidated, notes
