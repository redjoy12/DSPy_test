# PromptForge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a DSPy-based prompt engineering tool with two pipelines — CreatePrompt (from scratch) and IteratePrompt (modify existing) — with AI-as-judge evaluation, optional DSPy optimization, and file-based prompt versioning.

**Architecture:** Hybrid approach using DSPy ChainOfThought modules that work out of the box with zero training data (AI-as-judge), and optionally improve via DSPy optimizers (BootstrapFewShot/MIPROv2) when the user provides evaluation examples. File-based JSON versioning for prompt history.

**Tech Stack:** Python 3.11+, DSPy, OpenAI (via LiteLLM), pytest

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/pipelines/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/store/__init__.py`
- Create: `tests/__init__.py`
- Create: `prompts/.gitkeep`
- Create: `examples/.gitkeep`

**Step 1: Create requirements.txt**

```txt
dspy>=2.6.0
openai>=1.0.0
pytest>=8.0.0
python-dotenv>=1.0.0
```

**Step 2: Create package structure**

Create all `__init__.py` files as empty files. Create `prompts/.gitkeep` and `examples/.gitkeep` as empty files.

**Step 3: Create virtual environment and install dependencies**

Run:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```
Expected: All packages install successfully.

**Step 4: Verify DSPy imports**

Run: `python -c "import dspy; print(dspy.__version__)"`
Expected: Prints version >= 2.6.0

**Step 5: Commit**

```bash
git init
git add requirements.txt src/ tests/ prompts/.gitkeep examples/.gitkeep
git commit -m "chore: scaffold project structure and dependencies"
```

---

### Task 2: DSPy Configuration Module

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
from unittest.mock import patch

from src.config import configure_lm, get_default_model


def test_get_default_model_returns_string():
    model = get_default_model()
    assert isinstance(model, str)
    assert "openai" in model or "gpt" in model


def test_configure_lm_sets_dspy_config():
    """Verify configure_lm calls dspy.configure with an LM instance."""
    with patch("src.config.dspy") as mock_dspy:
        mock_dspy.LM.return_value = "fake_lm"
        configure_lm(model="openai/gpt-4o-mini")
        mock_dspy.LM.assert_called_once_with("openai/gpt-4o-mini", temperature=0.7, max_tokens=2000)
        mock_dspy.configure.assert_called_once_with(lm="fake_lm")


def test_configure_lm_custom_params():
    """Verify custom temperature and max_tokens are passed through."""
    with patch("src.config.dspy") as mock_dspy:
        mock_dspy.LM.return_value = "fake_lm"
        configure_lm(model="openai/gpt-4o", temperature=0.3, max_tokens=500)
        mock_dspy.LM.assert_called_once_with("openai/gpt-4o", temperature=0.3, max_tokens=500)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

**Step 3: Write minimal implementation**

```python
# src/config.py
import dspy


DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_default_model() -> str:
    return DEFAULT_MODEL


def configure_lm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> dspy.LM:
    lm = dspy.LM(model, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add DSPy LM configuration module"
```

---

### Task 3: Prompt Store — File-Based Versioning

**Files:**
- Create: `src/store/prompt_store.py`
- Create: `tests/test_prompt_store.py`

**Step 1: Write the failing tests**

```python
# tests/test_prompt_store.py
import json
import pytest
from pathlib import Path

from src.store.prompt_store import PromptStore, PromptVersion


@pytest.fixture
def store(tmp_path):
    return PromptStore(base_dir=tmp_path)


class TestPromptVersion:
    def test_create_prompt_version(self):
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="You are a helpful assistant.",
            description="General assistant",
            change_request=None,
            changes_made=None,
            quality_score=0.85,
            judge_feedback="Clear and concise",
            pipeline="create",
            model="openai/gpt-4o-mini",
        )
        assert pv.version == 1
        assert pv.parent_version is None
        assert pv.prompt_text == "You are a helpful assistant."

    def test_prompt_version_to_dict(self):
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="Test prompt",
            description="Test",
            quality_score=0.9,
            judge_feedback="Good",
            pipeline="create",
            model="openai/gpt-4o-mini",
        )
        d = pv.to_dict()
        assert d["version"] == 1
        assert d["prompt_text"] == "Test prompt"
        assert "timestamp" in d

    def test_prompt_version_from_dict(self):
        data = {
            "version": 2,
            "parent_version": 1,
            "prompt_text": "Updated prompt",
            "description": "Test",
            "change_request": "Make it shorter",
            "changes_made": "Removed filler words",
            "quality_score": 0.92,
            "judge_feedback": "More concise",
            "timestamp": "2026-03-25T10:00:00Z",
            "metadata": {"pipeline": "iterate", "model": "openai/gpt-4o-mini"},
        }
        pv = PromptVersion.from_dict(data)
        assert pv.version == 2
        assert pv.parent_version == 1
        assert pv.change_request == "Make it shorter"


class TestPromptStore:
    def test_save_first_version(self, store, tmp_path):
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="You are a helpful assistant.",
            description="General assistant",
            quality_score=0.85,
            judge_feedback="Good",
            pipeline="create",
            model="openai/gpt-4o-mini",
        )
        store.save("my_prompt", pv)
        path = tmp_path / "my_prompt" / "v1.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1

    def test_load_version(self, store):
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="Test prompt",
            description="Test",
            quality_score=0.9,
            judge_feedback="Good",
            pipeline="create",
            model="openai/gpt-4o-mini",
        )
        store.save("test_prompt", pv)
        loaded = store.load("test_prompt", version=1)
        assert loaded.prompt_text == "Test prompt"
        assert loaded.version == 1

    def test_load_latest_version(self, store):
        for i in range(1, 4):
            pv = PromptVersion(
                version=i,
                parent_version=i - 1 if i > 1 else None,
                prompt_text=f"Prompt v{i}",
                description="Test",
                quality_score=0.8 + i * 0.01,
                judge_feedback="Good",
                pipeline="iterate" if i > 1 else "create",
                model="openai/gpt-4o-mini",
            )
            store.save("evolving_prompt", pv)
        latest = store.load_latest("evolving_prompt")
        assert latest.version == 3
        assert latest.prompt_text == "Prompt v3"

    def test_get_next_version_number(self, store):
        assert store.get_next_version("new_prompt") == 1
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="First",
            description="Test",
            quality_score=0.8,
            judge_feedback="OK",
            pipeline="create",
            model="openai/gpt-4o-mini",
        )
        store.save("new_prompt", pv)
        assert store.get_next_version("new_prompt") == 2

    def test_list_versions(self, store):
        for i in range(1, 3):
            pv = PromptVersion(
                version=i,
                parent_version=i - 1 if i > 1 else None,
                prompt_text=f"Prompt v{i}",
                description="Test",
                quality_score=0.8,
                judge_feedback="OK",
                pipeline="create",
                model="openai/gpt-4o-mini",
            )
            store.save("listed_prompt", pv)
        versions = store.list_versions("listed_prompt")
        assert versions == [1, 2]

    def test_list_prompts(self, store):
        for name in ["alpha", "beta"]:
            pv = PromptVersion(
                version=1,
                parent_version=None,
                prompt_text=f"{name} prompt",
                description=name,
                quality_score=0.8,
                judge_feedback="OK",
                pipeline="create",
                model="openai/gpt-4o-mini",
            )
            store.save(name, pv)
        prompts = store.list_prompts()
        assert set(prompts) == {"alpha", "beta"}

    def test_load_nonexistent_prompt_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent", version=1)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prompt_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/store/prompt_store.py
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class PromptVersion:
    version: int
    parent_version: Optional[int]
    prompt_text: str
    description: str
    quality_score: float
    judge_feedback: str
    pipeline: str
    model: str
    change_request: Optional[str] = None
    changes_made: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "prompt_text": self.prompt_text,
            "description": self.description,
            "change_request": self.change_request,
            "changes_made": self.changes_made,
            "quality_score": self.quality_score,
            "judge_feedback": self.judge_feedback,
            "timestamp": self.timestamp,
            "metadata": {
                "pipeline": self.pipeline,
                "model": self.model,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptVersion":
        metadata = data.get("metadata", {})
        return cls(
            version=data["version"],
            parent_version=data.get("parent_version"),
            prompt_text=data["prompt_text"],
            description=data.get("description", ""),
            change_request=data.get("change_request"),
            changes_made=data.get("changes_made"),
            quality_score=data.get("quality_score", 0.0),
            judge_feedback=data.get("judge_feedback", ""),
            pipeline=metadata.get("pipeline", "unknown"),
            model=metadata.get("model", "unknown"),
            timestamp=data.get("timestamp", ""),
        )


class PromptStore:
    def __init__(self, base_dir: Path | str = "prompts"):
        self.base_dir = Path(base_dir)

    def save(self, name: str, version: PromptVersion) -> Path:
        prompt_dir = self.base_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"v{version.version}.json"
        path.write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, name: str, version: int) -> PromptVersion:
        path = self.base_dir / name / f"v{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt '{name}' version {version} not found at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return PromptVersion.from_dict(data)

    def load_latest(self, name: str) -> PromptVersion:
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for prompt '{name}'")
        return self.load(name, max(versions))

    def get_next_version(self, name: str) -> int:
        versions = self.list_versions(name)
        return max(versions) + 1 if versions else 1

    def list_versions(self, name: str) -> list[int]:
        prompt_dir = self.base_dir / name
        if not prompt_dir.exists():
            return []
        versions = []
        for f in prompt_dir.glob("v*.json"):
            try:
                versions.append(int(f.stem[1:]))
            except ValueError:
                continue
        return sorted(versions)

    def list_prompts(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and any(d.glob("v*.json"))
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prompt_store.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/store/prompt_store.py tests/test_prompt_store.py
git commit -m "feat: add file-based prompt versioning store"
```

---

### Task 4: AI-as-Judge Evaluation Metric

**Files:**
- Create: `src/evaluation/judge.py`
- Create: `tests/test_judge.py`

**Step 1: Write the failing tests**

```python
# tests/test_judge.py
import pytest
from unittest.mock import MagicMock

from src.evaluation.judge import PromptQualityJudge, prompt_quality_metric, prompt_comparison_metric


class TestPromptQualityJudge:
    def test_judge_signature_has_correct_fields(self):
        """Verify the signature class has the expected input/output fields."""
        from src.evaluation.judge import PromptQualitySignature
        input_fields = PromptQualitySignature.input_fields
        output_fields = PromptQualitySignature.output_fields
        assert "prompt_text" in input_fields
        assert "original_description" in input_fields
        assert "quality_score" in output_fields
        assert "feedback" in output_fields

    def test_comparison_signature_has_correct_fields(self):
        from src.evaluation.judge import PromptComparisonSignature
        input_fields = PromptComparisonSignature.input_fields
        output_fields = PromptComparisonSignature.output_fields
        assert "original_prompt" in input_fields
        assert "improved_prompt" in input_fields
        assert "change_request" in input_fields
        assert "improvement_score" in output_fields
        assert "feedback" in output_fields


class TestPromptQualityMetric:
    def test_quality_metric_returns_float(self):
        """Test that the metric function returns a float score."""
        mock_judge = MagicMock()
        mock_judge.return_value = MagicMock(quality_score=0.85, feedback="Good prompt")
        judge = PromptQualityJudge(judge_module=mock_judge)
        score, feedback = judge.evaluate_quality(
            prompt_text="You are a helpful assistant.",
            description="General assistant",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(feedback, str)

    def test_comparison_metric_returns_float(self):
        mock_judge = MagicMock()
        mock_judge.return_value = MagicMock(improvement_score=0.9, feedback="Better")
        judge = PromptQualityJudge(comparison_module=mock_judge)
        score, feedback = judge.evaluate_comparison(
            original_prompt="Old prompt",
            improved_prompt="New prompt",
            change_request="Make it better",
        )
        assert isinstance(score, float)
        assert isinstance(feedback, str)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/evaluation/judge.py
import dspy


class PromptQualitySignature(dspy.Signature):
    """Score a prompt's quality on clarity, completeness, specificity, and best practices adherence."""
    prompt_text: str = dspy.InputField(desc="the prompt to evaluate")
    original_description: str = dspy.InputField(desc="what the prompt was supposed to do")
    quality_score: float = dspy.OutputField(desc="quality score from 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="specific suggestions for improvement")


class PromptComparisonSignature(dspy.Signature):
    """Compare an improved prompt against the original and the requested change."""
    original_prompt: str = dspy.InputField(desc="the original prompt before changes")
    improved_prompt: str = dspy.InputField(desc="the modified prompt after changes")
    change_request: str = dspy.InputField(desc="what changes were requested")
    improvement_score: float = dspy.OutputField(desc="how well the changes address the request, 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="assessment of the changes made")


class PromptQualityJudge:
    def __init__(
        self,
        judge_module=None,
        comparison_module=None,
    ):
        self.judge = judge_module or dspy.ChainOfThought(PromptQualitySignature)
        self.comparison = comparison_module or dspy.ChainOfThought(PromptComparisonSignature)

    def evaluate_quality(self, prompt_text: str, description: str) -> tuple[float, str]:
        result = self.judge(prompt_text=prompt_text, original_description=description)
        try:
            score = max(0.0, min(1.0, float(result.quality_score)))
        except (ValueError, TypeError):
            score = 0.0
        return score, result.feedback

    def evaluate_comparison(
        self,
        original_prompt: str,
        improved_prompt: str,
        change_request: str,
    ) -> tuple[float, str]:
        result = self.comparison(
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            change_request=change_request,
        )
        try:
            score = max(0.0, min(1.0, float(result.improvement_score)))
        except (ValueError, TypeError):
            score = 0.0
        return score, result.feedback


QUALITY_THRESHOLD = 0.7


def make_quality_metric(judge=None):
    """Factory for DSPy-compatible prompt quality metric."""
    judge = judge or PromptQualityJudge()
    def metric(example, pred, trace=None) -> float:
        score, _ = judge.evaluate_quality(
            prompt_text=pred.prompt_text,
            description=example.description,
        )
        if trace is not None:
            return score >= QUALITY_THRESHOLD
        return score
    return metric


def make_comparison_metric(judge=None):
    """Factory for DSPy-compatible prompt comparison metric."""
    judge = judge or PromptQualityJudge()
    def metric(example, pred, trace=None) -> float:
        score, _ = judge.evaluate_comparison(
            original_prompt=example.current_prompt,
            improved_prompt=pred.improved_prompt,
            change_request=example.change_request,
        )
        if trace is not None:
            return score >= QUALITY_THRESHOLD
        return score
    return metric


# Convenience aliases for backward compatibility
def prompt_quality_metric(example, pred, trace=None) -> float:
    return make_quality_metric()(example, pred, trace)


def prompt_comparison_metric(example, pred, trace=None) -> float:
    return make_comparison_metric()(example, pred, trace)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_judge.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/evaluation/judge.py tests/test_judge.py
git commit -m "feat: add AI-as-judge evaluation metrics"
```

---

### Task 5: Example-Based Evaluation Metric

**Files:**
- Create: `src/evaluation/example_metric.py`
- Create: `tests/test_example_metric.py`

**Step 1: Write the failing tests**

```python
# tests/test_example_metric.py
import pytest
from unittest.mock import MagicMock

from src.evaluation.example_metric import ExampleBasedMetric


class TestExampleBasedMetric:
    def test_perfect_score_when_all_examples_pass(self):
        """When the LLM produces expected outputs for all examples, score is 1.0."""
        mock_predict = MagicMock()
        mock_predict.side_effect = [
            MagicMock(output="Hello! How can I help you?"),
            MagicMock(output="Sure, I can help with that."),
        ]
        examples = [
            {"input": "Hi", "expected_output": "Hello! How can I help you?"},
            {"input": "Can you help?", "expected_output": "Sure, I can help with that."},
        ]
        metric = ExampleBasedMetric(predict_module=mock_predict)
        score = metric.evaluate("You are helpful.", examples)
        assert score == 1.0

    def test_partial_score(self):
        mock_predict = MagicMock()
        mock_predict.side_effect = [
            MagicMock(output="Hello! How can I help you?"),
            MagicMock(output="Wrong output"),
        ]
        examples = [
            {"input": "Hi", "expected_output": "Hello! How can I help you?"},
            {"input": "Can you help?", "expected_output": "Sure, I can help with that."},
        ]
        metric = ExampleBasedMetric(predict_module=mock_predict)
        score = metric.evaluate("You are helpful.", examples)
        assert score == 0.5

    def test_zero_score_when_no_examples_match(self):
        mock_predict = MagicMock()
        mock_predict.side_effect = [MagicMock(output="Wrong")]
        examples = [{"input": "Hi", "expected_output": "Hello"}]
        metric = ExampleBasedMetric(predict_module=mock_predict)
        score = metric.evaluate("Bad prompt", examples)
        assert score == 0.0

    def test_empty_examples_returns_zero(self):
        metric = ExampleBasedMetric()
        score = metric.evaluate("Some prompt", [])
        assert score == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_example_metric.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/evaluation/example_metric.py
import dspy


class TestPromptSignature(dspy.Signature):
    """Execute a prompt against a given input and produce the output."""
    system_prompt: str = dspy.InputField(desc="the system prompt to test")
    user_input: str = dspy.InputField(desc="the user input to process")
    output: str = dspy.OutputField(desc="the response following the system prompt's instructions")


class ExampleBasedMetric:
    def __init__(self, predict_module=None):
        self.predict = predict_module or dspy.Predict(TestPromptSignature)

    def evaluate(self, prompt_text: str, examples: list[dict]) -> float:
        if not examples:
            return 0.0

        passed = 0
        for ex in examples:
            try:
                result = self.predict(
                    system_prompt=prompt_text,
                    user_input=ex["input"],
                )
                if result.output.strip().lower() == ex["expected_output"].strip().lower():
                    passed += 1
            except Exception:
                pass  # Failed prediction counts as non-passing

        return passed / len(examples)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_example_metric.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/evaluation/example_metric.py tests/test_example_metric.py
git commit -m "feat: add example-based evaluation metric"
```

---

### Task 6: CreatePrompt Pipeline (Pipeline 1)

**Files:**
- Create: `src/pipelines/create_prompt.py`
- Create: `tests/test_create_prompt.py`

**Step 1: Write the failing tests**

```python
# tests/test_create_prompt.py
import pytest
from unittest.mock import MagicMock

from src.pipelines.create_prompt import CreatePromptSignature, CreatePromptPipeline


class TestCreatePromptSignature:
    def test_signature_has_correct_input_fields(self):
        input_fields = CreatePromptSignature.input_fields
        assert "description" in input_fields
        assert "context" in input_fields

    def test_signature_has_correct_output_fields(self):
        output_fields = CreatePromptSignature.output_fields
        assert "prompt_text" in output_fields
        assert "reasoning" in output_fields


class TestCreatePromptPipeline:
    def test_forward_returns_prediction_with_prompt_text(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(
            prompt_text="You are a customer support agent...",
            reasoning="Structured with role, tone, and constraints.",
        )
        pipeline = CreatePromptPipeline(generate_module=mock_cot)
        result = pipeline(description="A customer support chatbot")
        assert hasattr(result, "prompt_text")
        assert "customer support" in result.prompt_text.lower()

    def test_forward_passes_description_and_context(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(prompt_text="test", reasoning="test")
        pipeline = CreatePromptPipeline(generate_module=mock_cot)
        pipeline(description="Email classifier", context="For a SaaS product")
        mock_cot.assert_called_once_with(
            description="Email classifier",
            context="For a SaaS product",
        )

    def test_forward_uses_empty_context_by_default(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(prompt_text="test", reasoning="test")
        pipeline = CreatePromptPipeline(generate_module=mock_cot)
        pipeline(description="Code reviewer")
        mock_cot.assert_called_once_with(
            description="Code reviewer",
            context="",
        )

    def test_create_and_save_stores_versioned_prompt(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(
            prompt_text="You are a code reviewer.",
            reasoning="Direct role assignment.",
        )
        mock_judge = MagicMock()
        mock_judge.evaluate_quality.return_value = (0.88, "Good structure")
        mock_store = MagicMock()
        mock_store.get_next_version.return_value = 1

        pipeline = CreatePromptPipeline(
            generate_module=mock_cot,
            judge=mock_judge,
            store=mock_store,
        )
        result = pipeline.create_and_save(
            name="code_reviewer",
            description="A code review assistant",
        )
        assert result.prompt_text == "You are a code reviewer."
        mock_store.save.assert_called_once()
        saved_version = mock_store.save.call_args[0][1]
        assert saved_version.quality_score == 0.88
        assert saved_version.pipeline == "create"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_create_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/pipelines/create_prompt.py
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
        super().__init__()  # Required: registers sub-modules for DSPy optimization
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_create_prompt.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/pipelines/create_prompt.py tests/test_create_prompt.py
git commit -m "feat: add CreatePrompt pipeline"
```

---

### Task 7: IteratePrompt Pipeline (Pipeline 2)

**Files:**
- Create: `src/pipelines/iterate_prompt.py`
- Create: `tests/test_iterate_prompt.py`

**Step 1: Write the failing tests**

```python
# tests/test_iterate_prompt.py
from unittest.mock import MagicMock

from src.pipelines.iterate_prompt import IteratePromptSignature, IteratePromptPipeline


class TestIteratePromptSignature:
    def test_signature_has_correct_input_fields(self):
        input_fields = IteratePromptSignature.input_fields
        assert "current_prompt" in input_fields
        assert "change_request" in input_fields
        assert "failing_examples" in input_fields

    def test_signature_has_correct_output_fields(self):
        output_fields = IteratePromptSignature.output_fields
        assert "improved_prompt" in output_fields
        assert "changes_made" in output_fields


class TestIteratePromptPipeline:
    def test_forward_returns_improved_prompt(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(
            improved_prompt="You are a concise assistant...",
            changes_made="Removed verbose instructions.",
        )
        pipeline = IteratePromptPipeline(generate_module=mock_cot)
        result = pipeline(
            current_prompt="You are a very verbose assistant...",
            change_request="Make it more concise",
        )
        assert hasattr(result, "improved_prompt")
        assert hasattr(result, "changes_made")

    def test_forward_passes_all_fields(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(improved_prompt="test", changes_made="test")
        pipeline = IteratePromptPipeline(generate_module=mock_cot)
        pipeline(
            current_prompt="Old prompt",
            change_request="Add error handling",
            failing_examples="Input: error case -> Expected: graceful response",
        )
        mock_cot.assert_called_once_with(
            current_prompt="Old prompt",
            change_request="Add error handling",
            failing_examples="Input: error case -> Expected: graceful response",
        )

    def test_forward_uses_empty_failing_examples_by_default(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(improved_prompt="test", changes_made="test")
        pipeline = IteratePromptPipeline(generate_module=mock_cot)
        pipeline(current_prompt="Old prompt", change_request="Improve tone")
        mock_cot.assert_called_once_with(
            current_prompt="Old prompt",
            change_request="Improve tone",
            failing_examples="",
        )

    def test_iterate_and_save_stores_versioned_prompt(self):
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(
            improved_prompt="Improved prompt text.",
            changes_made="Added error handling behavior.",
        )
        mock_judge = MagicMock()
        mock_judge.evaluate_comparison.return_value = (0.92, "Good improvement")
        mock_store = MagicMock()
        mock_store.get_next_version.return_value = 3
        mock_store.load_latest.return_value = MagicMock(
            prompt_text="Old prompt text.",
            version=2,
            description="Test prompt",
        )

        pipeline = IteratePromptPipeline(
            generate_module=mock_cot,
            judge=mock_judge,
            store=mock_store,
        )
        result = pipeline.iterate_and_save(
            name="my_prompt",
            change_request="Add error handling",
        )
        assert result.prompt_text == "Improved prompt text."
        mock_store.save.assert_called_once()
        saved_version = mock_store.save.call_args[0][1]
        assert saved_version.version == 3
        assert saved_version.parent_version == 2
        assert saved_version.change_request == "Add error handling"
        assert saved_version.pipeline == "iterate"

    def test_iterate_and_save_with_explicit_prompt(self):
        """When passing current_prompt directly instead of loading from store."""
        mock_cot = MagicMock()
        mock_cot.return_value = MagicMock(
            improved_prompt="Better prompt.",
            changes_made="Improved clarity.",
        )
        mock_judge = MagicMock()
        mock_judge.evaluate_comparison.return_value = (0.85, "OK")
        mock_store = MagicMock()
        mock_store.get_next_version.return_value = 1
        mock_store.list_versions.return_value = []

        pipeline = IteratePromptPipeline(
            generate_module=mock_cot,
            judge=mock_judge,
            store=mock_store,
        )
        result = pipeline.iterate_and_save(
            name="new_prompt",
            change_request="Improve it",
            current_prompt="Original prompt.",
            description="A test prompt",
        )
        assert result.prompt_text == "Better prompt."
        mock_store.load_latest.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_iterate_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/pipelines/iterate_prompt.py
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
        super().__init__()  # Required: registers sub-modules for DSPy optimization
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
        model: str = "openai/gpt-4o-mini",
    ) -> PromptVersion:
        if current_prompt is None:
            latest = self.store.load_latest(name)
            current_prompt = latest.prompt_text
            parent_version = latest.version
            description = description or latest.description
        else:
            existing = self.store.list_versions(name)
            parent_version = max(existing) if existing else None
            description = description or ""

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
            model=model,
        )
        self.store.save(name, version)
        return version
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_iterate_prompt.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/pipelines/iterate_prompt.py tests/test_iterate_prompt.py
git commit -m "feat: add IteratePrompt pipeline"
```

---

### Task 8: Optimizer Runner Utility

**Files:**
- Create: `src/optimizer.py`
- Create: `tests/test_optimizer.py`

**Step 1: Write the failing tests**

```python
# tests/test_optimizer.py
import pytest
from unittest.mock import MagicMock, patch

from src.optimizer import OptimizerRunner


class TestOptimizerRunner:
    def test_select_optimizer_bootstrap_for_small_data(self):
        runner = OptimizerRunner()
        optimizer_cls = runner.select_optimizer(num_examples=10)
        assert optimizer_cls.__name__ == "BootstrapFewShot"

    def test_select_optimizer_mipro_for_large_data(self):
        runner = OptimizerRunner()
        optimizer_cls = runner.select_optimizer(num_examples=200)
        assert optimizer_cls.__name__ == "MIPROv2"

    def test_select_optimizer_explicit_override(self):
        runner = OptimizerRunner()
        optimizer_cls = runner.select_optimizer(num_examples=10, optimizer_name="MIPROv2")
        assert optimizer_cls.__name__ == "MIPROv2"

    @patch("src.optimizer.dspy")
    def test_optimize_calls_compile(self, mock_dspy):
        mock_program = MagicMock()
        mock_optimized = MagicMock()
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.compile.return_value = mock_optimized
        mock_dspy.BootstrapFewShot.return_value = mock_optimizer_instance

        runner = OptimizerRunner()
        result = runner.optimize(
            program=mock_program,
            trainset=[MagicMock()] * 10,
            metric=lambda ex, pred, trace=None: True,
        )
        assert result == mock_optimized
        mock_optimizer_instance.compile.assert_called_once()

    @patch("src.optimizer.dspy")
    def test_optimize_saves_when_path_given(self, mock_dspy):
        mock_program = MagicMock()
        mock_optimized = MagicMock()
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.compile.return_value = mock_optimized
        mock_dspy.BootstrapFewShot.return_value = mock_optimizer_instance

        runner = OptimizerRunner()
        runner.optimize(
            program=mock_program,
            trainset=[MagicMock()] * 10,
            metric=lambda ex, pred, trace=None: True,
            save_path="optimized.json",
        )
        mock_optimized.save.assert_called_once_with("optimized.json")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/optimizer.py
from typing import Callable

import dspy


class OptimizerRunner:
    OPTIMIZER_THRESHOLD = 50  # Below this, use BootstrapFewShot; above, use MIPROv2

    def select_optimizer(
        self,
        num_examples: int,
        optimizer_name: str | None = None,
    ) -> type:
        if optimizer_name:
            try:
                cls = getattr(dspy, optimizer_name)
            except AttributeError:
                raise ValueError(
                    f"Unknown optimizer '{optimizer_name}'. "
                    f"Must be a valid dspy optimizer class (e.g. 'BootstrapFewShot', 'MIPROv2')."
                )
            return cls
        if num_examples < self.OPTIMIZER_THRESHOLD:
            return dspy.BootstrapFewShot
        return dspy.MIPROv2

    def optimize(
        self,
        program: dspy.Module,
        trainset: list,
        metric: Callable,
        optimizer_name: str | None = None,
        save_path: str | None = None,
        **kwargs,
    ) -> dspy.Module:
        optimizer_cls = self.select_optimizer(
            num_examples=len(trainset),
            optimizer_name=optimizer_name,
        )

        cls_name = getattr(optimizer_cls, '__name__', '')
        if cls_name == "MIPROv2":
            optimizer = optimizer_cls(metric=metric, auto="medium", **kwargs)
        elif cls_name == "BootstrapFewShot":
            optimizer = optimizer_cls(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
                **kwargs,
            )
        else:
            optimizer = optimizer_cls(metric=metric, **kwargs)

        optimized = optimizer.compile(program, trainset=trainset)

        if save_path:
            optimized.save(save_path)

        return optimized
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/optimizer.py tests/test_optimizer.py
git commit -m "feat: add optimizer runner utility"
```

---

### Task 9: Usage Examples

**Files:**
- Create: `examples/create_example.py`
- Create: `examples/iterate_example.py`

**Step 1: Write create_example.py**

```python
# examples/create_example.py
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
```

**Step 2: Write iterate_example.py**

```python
# examples/iterate_example.py
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
```

**Step 3: Commit**

```bash
git add examples/create_example.py examples/iterate_example.py
git commit -m "docs: add usage examples for both pipelines"
```

---

### Task 10: Integration Test — Full Round-Trip

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""
Integration test for the full create -> iterate round-trip.
Uses mocked LLM calls so no API key needed.
"""
import pytest
from unittest.mock import MagicMock

from src.pipelines.create_prompt import CreatePromptPipeline
from src.pipelines.iterate_prompt import IteratePromptPipeline
from src.evaluation.judge import PromptQualityJudge
from src.store.prompt_store import PromptStore


@pytest.fixture
def store(tmp_path):
    return PromptStore(base_dir=tmp_path)


def make_mock_judge(quality_score=0.85, comparison_score=0.90):
    judge = MagicMock(spec=PromptQualityJudge)
    judge.evaluate_quality.return_value = (quality_score, "Good prompt")
    judge.evaluate_comparison.return_value = (comparison_score, "Good improvement")
    return judge


class TestFullRoundTrip:
    def test_create_then_iterate(self, store):
        # Mock the create module
        mock_create = MagicMock()
        mock_create.return_value = MagicMock(
            prompt_text="You are a helpful coding assistant. You help users write clean Python code.",
            reasoning="Role-based prompt with specific domain.",
        )

        # Mock the iterate module
        mock_iterate = MagicMock()
        mock_iterate.return_value = MagicMock(
            improved_prompt="You are a helpful coding assistant. You help users write clean Python code. "
                           "When you encounter errors, explain them clearly and suggest fixes.",
            changes_made="Added error explanation behavior.",
        )

        judge = make_mock_judge()

        # Step 1: Create
        create_pipeline = CreatePromptPipeline(
            generate_module=mock_create, judge=judge, store=store,
        )
        v1 = create_pipeline.create_and_save(
            name="coding_assistant",
            description="A coding assistant for Python developers",
        )
        assert v1.version == 1
        assert v1.parent_version is None
        assert v1.pipeline == "create"
        assert "coding assistant" in v1.prompt_text.lower()

        # Step 2: Iterate
        iterate_pipeline = IteratePromptPipeline(
            generate_module=mock_iterate, judge=judge, store=store,
        )
        v2 = iterate_pipeline.iterate_and_save(
            name="coding_assistant",
            change_request="Add error explanation behavior",
        )
        assert v2.version == 2
        assert v2.parent_version == 1
        assert v2.pipeline == "iterate"
        assert v2.change_request == "Add error explanation behavior"

        # Step 3: Verify lineage
        all_versions = store.list_versions("coding_assistant")
        assert all_versions == [1, 2]

        loaded_v1 = store.load("coding_assistant", 1)
        loaded_v2 = store.load("coding_assistant", 2)
        assert loaded_v2.parent_version == loaded_v1.version

    def test_multiple_iterations_build_lineage(self, store):
        judge = make_mock_judge()

        mock_create = MagicMock()
        mock_create.return_value = MagicMock(prompt_text="V1 prompt", reasoning="Initial")
        create_pipeline = CreatePromptPipeline(
            generate_module=mock_create, judge=judge, store=store,
        )
        create_pipeline.create_and_save(name="evolving", description="Test prompt")

        iterate_pipeline = IteratePromptPipeline(judge=judge, store=store)

        for i in range(2, 5):
            mock_iter = MagicMock()
            mock_iter.return_value = MagicMock(
                improved_prompt=f"V{i} prompt",
                changes_made=f"Change {i}",
            )
            iterate_pipeline.generate = mock_iter
            iterate_pipeline.iterate_and_save(
                name="evolving",
                change_request=f"Improvement {i}",
            )

        versions = store.list_versions("evolving")
        assert versions == [1, 2, 3, 4]

        latest = store.load_latest("evolving")
        assert latest.version == 4
        assert latest.parent_version == 3
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add full round-trip integration test"
```

---

### Task 11: Final — Run Full Test Suite & Verify

**Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 26 tests across 6 test files)

**Step 2: Verify project structure**

Run: `find . -type f -name "*.py" | sort`
Expected: All files from the plan exist in the correct locations.

**Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: project cleanup and final verification"
```

---

### Post-Review Plan Corrections (2026-03-28)

The following corrections were applied to this plan based on the verified code review (`docs/code reviews/2026-03-27-full-project-review.md`):

| Correction | Task | What changed |
|------------|------|-------------|
| Added `super().__init__()` | Task 6, Task 7 | Both `CreatePromptPipeline` and `IteratePromptPipeline` now call `super().__init__()` as the first line in `__init__`. Required for DSPy's `Module` to discover sub-modules for optimization/serialization. |
| Added `optimizer_name` validation | Task 8 | `select_optimizer` now wraps `getattr(dspy, optimizer_name)` in try/except and raises a descriptive `ValueError` for invalid names. |
| Removed unused `import pytest` | Task 2 | `tests/test_config.py` no longer imports `pytest` (it was never referenced). |

**Implementation deviations not caused by plan bugs** (plan was already correct, implementation diverged):
- **I-1**: Plan specifies wrapper functions for convenience aliases (`def prompt_quality_metric(...)`). Implementation used module-level singleton instances instead (`prompt_quality_metric = make_quality_metric()`), causing import-time side effects.
- **I-2**: Plan defines `QUALITY_THRESHOLD = 0.7` constant. Implementation hardcoded `0.7` in two places instead of referencing the constant.
