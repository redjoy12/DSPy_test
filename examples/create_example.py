"""
Example: Create a prompt from scratch using PromptForge.

Usage:
    export OPENAI_API_KEY=your-key-here
    python examples/create_example.py
"""
from src.config import configure_lm
from src.pipelines.create_prompt import CreatePromptPipeline
from src.store.prompt_store import PromptStore

if __name__ == "__main__":
    # 1. Configure DSPy with OpenAI
    configure_lm(model="openai/gpt-4o-mini")

    # 2. Initialize pipeline with a store
    store = PromptStore(base_dir="prompts")
    pipeline = CreatePromptPipeline(store=store)

    # 3. Create a prompt from a description
    version = pipeline.create_and_save(
        name="customer_support",
        description="A customer support chatbot for an e-commerce platform that handles returns, "
                    "order status inquiries, and shipping questions",
        context="Tone should be friendly but professional. Target audience is retail customers. "
                "The bot should always check order status before providing return instructions.",
    )

    print(f"Created prompt v{version.version}")
    print(f"Quality score: {version.quality_score:.2f}")
    print(f"Judge feedback: {version.judge_feedback}")
    print(f"\n--- Generated Prompt ---\n{version.prompt_text}")
