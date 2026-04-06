import json
import re
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
    structured_examples: Optional[list[dict]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "prompt_text": self.prompt_text,
            "description": self.description,
            "change_request": self.change_request,
            "changes_made": self.changes_made,
            "structured_examples": self.structured_examples,
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
            structured_examples=data.get("structured_examples"),
            quality_score=data.get("quality_score", 0.0),
            judge_feedback=data.get("judge_feedback", ""),
            pipeline=metadata.get("pipeline", "unknown"),
            model=metadata.get("model", "unknown"),
            timestamp=data.get("timestamp", ""),
        )

    @staticmethod
    def format_examples_as_text(examples: Optional[list[dict]]) -> str:
        """Serialize structured failing examples into a readable text block.

        Each example is a dict of the form::

            {
                "messages": [
                    {"role": "human", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ],
                "unsatisfactory_output": "the final bad output from the assistant",
            }

        The returned string is meant to be fed into the ``failing_examples``
        string input field of :class:`IteratePromptSignature` so the LLM can
        read the conversation context and understand where the current system
        prompt is producing bad results.
        """
        if not examples:
            return ""

        blocks: list[str] = []
        for i, example in enumerate(examples, start=1):
            lines = [f"--- Example {i} ---"]
            for msg in example.get("messages", []) or []:
                role = str(msg.get("role", "")).strip().lower()
                content = str(msg.get("content", "")).strip()
                if role == "human":
                    label = "Human"
                elif role == "assistant":
                    label = "Assistant"
                else:
                    label = role.capitalize() or "Unknown"
                lines.append(f"{label}: {content}")
            unsatisfactory = str(example.get("unsatisfactory_output", "")).strip()
            if unsatisfactory:
                lines.append(f"Unsatisfactory Output: {unsatisfactory}")
            lines.append("---")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)


class PromptStore:
    """File-based prompt version storage.

    Note: This store assumes a single-writer process. Concurrent writers may
    produce conflicting version numbers. For production multi-process usage,
    add file locking or switch to a database-backed store.
    """

    def __init__(self, base_dir: Path | str = "prompts"):
        self.base_dir = Path(base_dir)

    @staticmethod
    def validate_name(name: str) -> None:
        """Reject prompt names that could escape the base directory."""
        if not name or not re.match(r'^[\w\-. ]+$', name):
            raise ValueError(
                "Prompt name may only contain letters, digits, hyphens, "
                "underscores, dots, and spaces."
            )
        if name.strip('. ') == '':
            raise ValueError("Prompt name cannot consist only of dots and spaces.")

    def save(self, name: str, version: PromptVersion) -> Path:
        self.validate_name(name)
        prompt_dir = self.base_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"v{version.version}.json"
        path.write_text(json.dumps(version.to_dict(), indent=2), encoding="utf-8")
        return path

    def load(self, name: str, version: int) -> PromptVersion:
        self.validate_name(name)
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
