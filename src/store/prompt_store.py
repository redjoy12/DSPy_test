import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HAS_FCNTL = sys.platform != "win32"

if _HAS_FCNTL:
    import fcntl

if not _HAS_FCNTL:
    logger.warning(
        "File locking is not available on this platform (Windows). "
        "Concurrent writes to the same prompt may cause data corruption. "
        "Consider using a networked lock service or database for production."
    )


def _acquire_lock(lock_file) -> bool:
    """Attempt to acquire file lock. Returns True if successful, False otherwise."""
    if not _HAS_FCNTL:
        return False
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return True
    except (AttributeError, OSError):
        return False


def _release_lock(lock_file) -> None:
    """Release file lock if held."""
    if not _HAS_FCNTL:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (AttributeError, OSError):
        pass


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
    abstracted_patterns: Optional[str] = None
    validation_passed: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "prompt_text": self.prompt_text,
            "description": self.description,
            "change_request": self.change_request,
            "changes_made": self.changes_made,
            "structured_examples": self.structured_examples,
            "abstracted_patterns": self.abstracted_patterns,
            "validation_passed": self.validation_passed,
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
            abstracted_patterns=data.get("abstracted_patterns"),
            validation_passed=data.get("validation_passed", True),
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
    """File-based prompt version storage with file locking for concurrent access.

    Uses file locking to prevent race conditions when multiple processes
    write to the same prompt. For cross-platform compatibility, locking
    is attempted but silently degrades if not available.
    """

    def __init__(self, base_dir: Path | str = "prompts"):
        self.base_dir = Path(base_dir).resolve()
        if ".." in str(self.base_dir):
            raise ValueError("base_dir cannot contain '..'")

    @staticmethod
    def validate_name(name: str) -> None:
        """Reject prompt names that could escape the base directory."""
        if not name or not re.match(r"^[\w\-. ]+$", name):
            raise ValueError(
                "Prompt name may only contain letters, digits, hyphens, "
                "underscores, dots, and spaces."
            )
        if name.strip(". ") == "":
            raise ValueError("Prompt name cannot consist only of dots and spaces.")

    def save(self, name: str, version: PromptVersion) -> Path:
        self.validate_name(name)
        prompt_dir = self.base_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"v{version.version}.json"
        lock_path = prompt_dir / ".lock"

        with open(lock_path, "w") as lock_file:
            lock_held = _acquire_lock(lock_file)
            try:
                path.write_text(
                    json.dumps(version.to_dict(), indent=2), encoding="utf-8"
                )
            finally:
                if lock_held:
                    _release_lock(lock_file)
        return path

    def load(self, name: str, version: int) -> PromptVersion:
        self.validate_name(name)
        path = self.base_dir / name / f"v{version}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt '{name}' version {version} not found at {path}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return PromptVersion.from_dict(data)

    def load_latest(self, name: str) -> PromptVersion:
        versions = self.list_versions(name)
        if not versions:
            raise FileNotFoundError(f"No versions found for prompt '{name}'")
        return self.load(name, max(versions))

    def get_next_version(self, name: str) -> int:
        self.validate_name(name)
        prompt_dir = self.base_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        lock_path = prompt_dir / ".lock"

        with open(lock_path, "w") as lock_file:
            lock_held = _acquire_lock(lock_file)
            try:
                versions = self.list_versions(name)
                return max(versions) + 1 if versions else 1
            finally:
                if lock_held:
                    _release_lock(lock_file)

    def get_and_save_version(
        self, name: str, version: PromptVersion
    ) -> tuple[int, Path]:
        """Atomically get next version number and save the prompt.

        This method holds the lock throughout both operations to prevent
        race conditions where two processes get the same version number.

        Returns:
            A tuple of (version_number, path_to_saved_file)
        """
        self.validate_name(name)
        prompt_dir = self.base_dir / name
        prompt_dir.mkdir(parents=True, exist_ok=True)
        lock_path = prompt_dir / ".lock"

        with open(lock_path, "w") as lock_file:
            lock_held = _acquire_lock(lock_file)
            try:
                versions = self.list_versions(name)
                next_version = max(versions) + 1 if versions else 1
                version.version = next_version

                path = prompt_dir / f"v{version.version}.json"
                path.write_text(
                    json.dumps(version.to_dict(), indent=2), encoding="utf-8"
                )
                return next_version, path
            finally:
                if lock_held:
                    _release_lock(lock_file)

    def list_versions(self, name: str) -> list[int]:
        self.validate_name(name)
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
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and any(d.glob("v*.json"))
        )
