"""
Example: Optimize a prompt-generation pipeline using DSPy optimizers.

OptimizerRunner selects the best DSPy optimizer (BootstrapFewShot for
small datasets, MIPROv2 for larger ones) and compiles your pipeline
against a training set and quality metric.

Usage:
    export OPENAI_API_KEY=your-key-here
    python examples/optimizer_example.py
"""
import dspy

from src.config import configure_lm
from src.optimizer import OptimizerRunner
from src.pipelines.create_prompt import CreatePromptPipeline
from src.evaluation.judge import make_quality_metric

if __name__ == "__main__":
    # 1. Configure DSPy with OpenAI
    configure_lm(model="openai/gpt-4o-mini")

    # 2. Build a training set of (description -> expected quality) examples
    trainset = [
        dspy.Example(
            description="A customer support chatbot for e-commerce",
        ).with_inputs("description"),
        dspy.Example(
            description="A Python code reviewer that catches bugs",
        ).with_inputs("description"),
        dspy.Example(
            description="A concise meeting-notes summarizer",
        ).with_inputs("description"),
    ]

    # 3. Create a DSPy-compatible quality metric
    metric = make_quality_metric()

    # 4. Select and run an optimizer
    runner = OptimizerRunner()
    pipeline = CreatePromptPipeline()

    # Let OptimizerRunner pick the best optimizer for our dataset size
    optimizer_cls = runner.select_optimizer(num_examples=len(trainset))
    print(f"Selected optimizer: {optimizer_cls.__name__}")

    # Compile the pipeline (this runs the optimizer training loop)
    optimized_pipeline = runner.optimize(
        program=pipeline,
        trainset=trainset,
        metric=metric,
        save_path="prompts/optimized_create_pipeline",
    )

    # 5. Use the optimized pipeline
    result = optimized_pipeline(description="A SQL query debugger")
    print(f"\n--- Optimized Prompt ---\n{result.prompt_text}")
