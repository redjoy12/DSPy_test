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
