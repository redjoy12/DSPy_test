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
