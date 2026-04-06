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


class TestStructuredExamples:
    """Tests for the multi-turn failing examples support on PromptVersion."""

    def _sample_examples(self) -> list[dict]:
        return [
            {
                "messages": [
                    {"role": "human", "content": "What products do you have?"},
                    {"role": "assistant", "content": "We have X, Y, Z"},
                    {"role": "human", "content": "Tell me more about X"},
                ],
                "unsatisfactory_output": "I don't know about X",
            },
            {
                "messages": [
                    {"role": "human", "content": "Hello"},
                ],
                "unsatisfactory_output": "Error: out of context",
            },
        ]

    def test_default_structured_examples_is_none(self):
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="p",
            description="d",
            quality_score=0.8,
            judge_feedback="ok",
            pipeline="create",
            model="m",
        )
        assert pv.structured_examples is None

    def test_structured_examples_roundtrip(self, store):
        examples = self._sample_examples()
        pv = PromptVersion(
            version=1,
            parent_version=None,
            prompt_text="Hello",
            description="test",
            quality_score=0.9,
            judge_feedback="ok",
            pipeline="iterate",
            model="m",
            structured_examples=examples,
        )
        store.save("struct_prompt", pv)
        loaded = store.load("struct_prompt", version=1)
        assert loaded.structured_examples == examples

    def test_structured_examples_absent_in_old_json_is_none(self, tmp_path, store):
        """Old JSON files without the key should deserialize with None."""
        prompt_dir = tmp_path / "legacy"
        prompt_dir.mkdir()
        # Write a legacy JSON file without the structured_examples key.
        (prompt_dir / "v1.json").write_text(
            json.dumps({
                "version": 1,
                "parent_version": None,
                "prompt_text": "Old prompt",
                "description": "legacy",
                "change_request": None,
                "changes_made": None,
                "quality_score": 0.7,
                "judge_feedback": "old",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"pipeline": "create", "model": "m"},
            }),
            encoding="utf-8",
        )
        loaded = store.load("legacy", version=1)
        assert loaded.structured_examples is None

    def test_format_examples_as_text_empty(self):
        assert PromptVersion.format_examples_as_text(None) == ""
        assert PromptVersion.format_examples_as_text([]) == ""

    def test_format_examples_as_text_renders_conversation(self):
        examples = self._sample_examples()
        text = PromptVersion.format_examples_as_text(examples)

        # Must contain headers for both examples
        assert "--- Example 1 ---" in text
        assert "--- Example 2 ---" in text
        # Capitalized role labels
        assert "Human: What products do you have?" in text
        assert "Assistant: We have X, Y, Z" in text
        assert "Human: Tell me more about X" in text
        # Unsatisfactory output line
        assert "Unsatisfactory Output: I don't know about X" in text
        assert "Unsatisfactory Output: Error: out of context" in text

    def test_format_examples_as_text_handles_missing_fields(self):
        """Examples without messages or unsatisfactory output still produce a block."""
        examples = [
            {"messages": [], "unsatisfactory_output": "bad"},
            {"messages": [{"role": "human", "content": "only human"}]},
        ]
        text = PromptVersion.format_examples_as_text(examples)
        assert "--- Example 1 ---" in text
        assert "Unsatisfactory Output: bad" in text
        assert "--- Example 2 ---" in text
        assert "Human: only human" in text
