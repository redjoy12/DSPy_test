import functools
import json
from pathlib import Path
from typing import Any


DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 4000


def get_config_path() -> Path:
    project_root = Path(__file__).parent.parent
    return project_root / "llm_config.json"


@functools.lru_cache(maxsize=1)
def load_llm_config() -> dict[str, Any]:
    config_path = get_config_path()

    if not config_path.exists():
        return {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

    temperature = config.get("temperature")
    if temperature is None or not isinstance(temperature, (int, float)):
        temperature = DEFAULT_TEMPERATURE
    else:
        temperature = max(0.0, min(2.0, float(temperature)))

    max_tokens = config.get("max_tokens")
    if max_tokens is None or not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS

    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def get_temperature() -> float:
    return load_llm_config()["temperature"]


def get_max_tokens() -> int:
    return load_llm_config()["max_tokens"]
