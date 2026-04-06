from __future__ import annotations
from typing import Optional

import dspy

from src.config_loader import get_temperature, get_max_tokens


DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_default_model() -> str:
    return DEFAULT_MODEL


def configure_lm(
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    temperature = temperature if temperature is not None else get_temperature()
    max_tokens = max_tokens if max_tokens is not None else get_max_tokens()
    lm = dspy.LM(model, temperature=temperature, max_tokens=max_tokens)
    dspy.configure(lm=lm)
    return lm
