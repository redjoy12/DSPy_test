import dspy


class ConsolidatePromptSignature(dspy.Signature):
    """Clean and consolidate an iterated prompt before it is saved.

    You are given TWO prompts:
      - ``original_prompt``: the prompt BEFORE the iteration step. This is the
        user's pre-existing content. Treat it as an untouchable BASELINE.
      - ``raw_prompt``: the prompt AFTER the iteration step. It equals
        ``original_prompt`` plus (possibly) new or reworded rules that the
        iterate step added to address the change request.

    Your job is to clean ONLY the delta — the content in ``raw_prompt`` that
    is NOT already in ``original_prompt``. Everything from the baseline must
    be preserved verbatim.

    === BASELINE ZONE (from original_prompt) — OFF-LIMITS ===

    Any rule, sentence, bullet, section header, or phrasing that is present in
    ``original_prompt`` is off-limits:

      - Preserve baseline wording EXACTLY. Do not rephrase, shorten, or
        "tighten" it.
      - Preserve baseline ordering, numbering, section structure, and voice.
      - Do NOT merge a baseline rule with a new rule.
      - Do NOT drop a baseline rule, even if a new rule overlaps with it.
      - If ``raw_prompt`` has reworded a baseline rule without the user's
        change_request explicitly asking for that change, RESTORE the original
        baseline wording.

    The only exception: if the user's ``change_request`` explicitly asked to
    modify a specific piece of baseline content (e.g. "make the tone more
    formal", "remove the bullet about X"), you may apply that specific change.
    When in doubt, preserve the baseline.

    === DELTA ZONE (new/reworded content only) — ELIGIBLE FOR CLEANING ===

    For content in ``raw_prompt`` that is NOT in ``original_prompt``, apply:

    1. Scenario-specific phrasing scrubbing. Strip any sentence that:
         - names a specific entity, product, brand, person, account, or place
         - cites specific numbers, dates, IDs, prices, or version strings
         - quotes a user/assistant utterance verbatim from a conversation log
         - uses anchors like "for example: ...", "e.g. when the user asks ...",
           "if the customer says ...", "such as the case where ..." that pin a
           rule to one specific scenario instead of stating it as a principle
       Replace such sentences with the equivalent general principle, or drop
       them entirely if a more general rule already covers them.

    2. Redundancy handling. Two cases:
         a) A new rule overlaps with a BASELINE rule → DROP the new rule.
            The baseline wins; the user already had that rule.
         b) Two NEW rules overlap with each other → MERGE them into a single
            consolidated new rule that preserves every distinct nuance from
            the originals, then add the merged rule in place of the overlap.

    === OUTPUT ===

    Return the full prompt: the verbatim baseline content, plus the cleaned
    delta content, assembled in a natural order. If the input ``raw_prompt``
    is already clean and non-redundant relative to ``original_prompt``, return
    it essentially unchanged. Never drop a new rule whose intent is not
    already covered by the baseline or another new rule.

    Concrete contrasts:

      Baseline contains: "Be concise."
      Raw prompt adds:   "Keep responses short and to the point."
      Output:            baseline "Be concise." is preserved verbatim; the
                         new rule is DROPPED (baseline already covers it).

      Baseline contains: (nothing about product pricing)
      Raw prompt adds:   "When the user asks about Acme Pro pricing, respond
                          with the current monthly cost from the pricing page."
      Output:            new rule is scrubbed to "When asked about product
                         pricing, look up the current pricing source before
                         answering." (baseline content untouched)
    """

    raw_prompt: str = dspy.InputField(
        desc="prompt produced by the iterate step (baseline + new/reworded rules); only the delta relative to original_prompt is eligible for cleaning"
    )
    original_prompt: str = dspy.InputField(
        desc="the prompt BEFORE the iterate step — the untouchable baseline whose rules and wording must be preserved verbatim in the output"
    )
    change_request: str = dspy.InputField(
        desc="the user's original change request, for context on what the new rules were trying to address and whether any baseline modification was explicitly requested"
    )
    abstracted_pattern: str = dspy.InputField(
        desc="optional: the abstracted failure pattern that triggered this iteration; use it to know which themes are legitimate vs. which are scenario detail"
    )
    consolidated_prompt: str = dspy.OutputField(
        desc="the full prompt with baseline content preserved verbatim and only the delta (new/reworded rules) scrubbed of scenario specifics and deduped"
    )
    consolidation_notes: str = dspy.OutputField(
        desc="brief summary of what was scrubbed, which new rules were merged or dropped, and which baseline rewordings (if any) were restored"
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
        original_prompt: str,
        change_request: str = "",
        abstracted_pattern: str = "",
    ) -> tuple[str, str]:
        """Run the consolidation pass.

        The consolidator is scoped to the DELTA: it cleans only the new or
        reworded content in ``raw_prompt`` that was not already present in
        ``original_prompt``. Baseline content from ``original_prompt`` is
        treated as off-limits and must be preserved verbatim in the output.
        This scoping is the core guarantee that iterating on an existing
        prompt does not inadvertently rewrite its pre-existing rules.

        Args:
            raw_prompt: The improved prompt produced by the iterate step
                (baseline + new/reworded rules).
            original_prompt: The pre-iteration prompt. Required — this is the
                baseline whose wording, ordering, and structure must be
                preserved verbatim. Only content in ``raw_prompt`` that is not
                in ``original_prompt`` is eligible for scrubbing or merging.
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
                original_prompt=original_prompt,
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
