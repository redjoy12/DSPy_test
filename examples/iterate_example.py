"""
Example: Iterate on an existing prompt using PromptForge.

Usage:
    export OPENAI_API_KEY=your-key-here
    python examples/create_example.py  # Run first to create initial prompt
    python examples/iterate_example.py
"""
from src.config import configure_lm
from src.pipelines.iterate_prompt import IteratePromptPipeline
from src.store.prompt_store import PromptStore

if __name__ == "__main__":
    # 1. Configure DSPy with OpenAI
    configure_lm(model="openai/gpt-4o-mini")

    # 2. Initialize pipeline with the same store
    store = PromptStore(base_dir="prompts")
    pipeline = IteratePromptPipeline(store=store)

    # 3. Iterate on existing prompt — add new behavior
    version = pipeline.iterate_and_save(
        name="customer_support",
        change_request="Add behavior for handling angry or frustrated customers. "
                       "The bot should acknowledge the customer's frustration, apologize, "
                       "and offer to escalate to a human agent if needed.",
    )

    print(f"Updated to v{version.version} (parent: v{version.parent_version})")
    print(f"Quality score: {version.quality_score:.2f}")
    print(f"Changes made: {version.changes_made}")
    print(f"\n--- Improved Prompt ---\n{version.prompt_text}")

    # 4. Another iteration — with failing examples
    version2 = pipeline.iterate_and_save(
        name="customer_support",
        change_request="Improve handling of refund amount questions",
        failing_examples=(
            "Input: 'How much will my refund be?' -> "
            "Expected: 'Let me look up your order to calculate the exact refund amount, "
            "including any applicable restocking fees.' "
            "Actual: 'Your refund will be processed soon.'"
        ),
    )

    print(f"\nUpdated to v{version2.version} (parent: v{version2.parent_version})")
    print(f"Quality score: {version2.quality_score:.2f}")
    print(f"Changes made: {version2.changes_made}")
    print(f"\n--- Improved Prompt ---\n{version2.prompt_text}")
