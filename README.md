# PromptForge

A DSPy-powered tool for generating, iterating, and versioning LLM prompts. PromptForge uses AI-as-judge evaluation to score prompt quality and maintains a full version history of every change.

## What it does

- **Create prompts from descriptions** — Generate system prompts using DSPy's `ChainOfThought`, scored automatically by an AI judge
- **Iterate with change requests** — Improve existing prompts by describing what to change, optionally providing failing examples
- **AI-as-judge evaluation** — Every prompt version is scored (0-1) on clarity, completeness, and specificity
- **File-based versioning** — Full lineage tracking with JSON files under `prompts/{name}/v{n}.json`
- **DSPy optimizer integration** — Auto-select and run `BootstrapFewShot` or `MIPROv2` to optimize your pipeline
- **Streamlit UI** — Browser-based interface for all of the above without writing code

## Requirements

- Python 3.10+
- An OpenAI API key

## Installation

```bash
git clone <repo-url>
cd DSPy_test

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your-key-here  # On Windows: set OPENAI_API_KEY=your-key-here
```

## Usage

### Option A — Streamlit UI

```bash
streamlit run app.py
```

Open the browser tab that appears. Enter your OpenAI API key in the sidebar and click **Connect** to load available models. The UI has four tabs:

| Tab | What it does |
|---|---|
| **Create** | Generate a new prompt from a name and description |
| **Iterate** | Improve an existing prompt with a change request |
| **Browse** | View all saved versions and compare diffs between them |
| **Optimize** | Run a DSPy optimizer against a JSON training set |

### Option B — Python API

**Create a prompt**

```python
from src.config import configure_lm
from src.pipelines.create_prompt import CreatePromptPipeline
from src.store.prompt_store import PromptStore

configure_lm(model="openai/gpt-4o-mini")

store = PromptStore(base_dir="prompts")
pipeline = CreatePromptPipeline(store=store)

version = pipeline.create_and_save(
    name="customer_support",
    description="A customer support chatbot for an e-commerce platform that handles returns, "
                "order status inquiries, and shipping questions",
    context="Tone should be friendly but professional. Target audience is retail customers.",
)

print(f"Created prompt v{version.version}")
print(f"Quality score: {version.quality_score:.2f}")
print(version.prompt_text)
```

**Iterate on a prompt**

```python
from src.pipelines.iterate_prompt import IteratePromptPipeline

pipeline = IteratePromptPipeline(store=store)

version = pipeline.iterate_and_save(
    name="customer_support",
    change_request="Add behavior for handling angry or frustrated customers. "
                   "The bot should acknowledge frustration, apologize, "
                   "and offer to escalate to a human agent if needed.",
)

print(f"Updated to v{version.version} (parent: v{version.parent_version})")
print(f"Quality score: {version.quality_score:.2f}")
```

You can also provide failing examples to guide the iteration:

```python
version = pipeline.iterate_and_save(
    name="customer_support",
    change_request="Improve handling of refund amount questions",
    failing_examples=(
        "Input: 'How much will my refund be?' -> "
        "Expected: 'Let me look up your order to calculate the exact refund amount.' "
        "Actual: 'Your refund will be processed soon.'"
    ),
)
```

**Optimize a pipeline with DSPy**

```python
import dspy
from src.optimizer import OptimizerRunner
from src.evaluation.judge import make_quality_metric

runner = OptimizerRunner()
pipeline = CreatePromptPipeline()

trainset = [
    dspy.Example(description="A customer support chatbot for e-commerce").with_inputs("description"),
    dspy.Example(description="A Python code reviewer that catches bugs").with_inputs("description"),
    dspy.Example(description="A concise meeting-notes summarizer").with_inputs("description"),
]

metric = make_quality_metric()

# Auto-selects BootstrapFewShot (<50 examples) or MIPROv2 (>=50)
optimized = runner.optimize(
    program=pipeline,
    trainset=trainset,
    metric=metric,
    save_path="prompts/optimized_create_pipeline",
)
```

**Enforce a minimum quality score**

```python
version = pipeline.create_and_save(
    name="my_prompt",
    description="...",
    min_score=0.7,  # Raises ValueError if judge scores below 0.7
)
```

## Running Tests

```bash
# Unit tests (no API key needed — uses DummyLM)
pytest

# Integration / smoke tests (requires OPENAI_API_KEY)
pytest tests/test_smoke.py -m integration
```

## Project Structure

```
app.py                           # Streamlit UI
src/
  config.py                      # configure_lm() — sets up DSPy with OpenAI
  optimizer.py                   # OptimizerRunner — auto-select and run DSPy optimizers
  pipelines/
    create_prompt.py             # CreatePromptPipeline
    iterate_prompt.py            # IteratePromptPipeline
  evaluation/
    judge.py                     # PromptQualityJudge — AI-as-judge scoring
    example_metric.py            # ExampleBasedMetric — token-overlap scoring
  store/
    prompt_store.py              # PromptStore — file-based JSON versioning
examples/                        # Standalone usage examples
tests/                           # Unit and integration tests
prompts/                         # Generated prompt versions (gitignored)
```
