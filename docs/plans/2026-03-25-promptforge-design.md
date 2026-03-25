# PromptForge — Design Document

**Date:** 2026-03-25
**Status:** Approved

## Overview

PromptForge is a Python developer tool built on DSPy that provides two pipelines for prompt engineering:

1. **CreatePrompt** — generates prompts from scratch given a natural language description
2. **IteratePrompt** — iterates on existing prompts by adding/modifying behaviors based on change requests and optional failing examples

The tool targets OpenAI models (GPT-4o, etc.), produces single self-contained prompts, and uses a hybrid approach: works immediately with zero training data via AI-as-judge evaluation, and gets better over time when the user provides evaluation examples and runs DSPy optimizers.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  PromptForge                     │
│              (Main Python Package)               │
├─────────────┬───────────────┬───────────────────┤
│  Pipeline 1 │  Pipeline 2   │    Shared Layer   │
│ CreatePrompt│ IteratePrompt │                   │
│             │               │  - AI Judge       │
│ description │ current_prompt│  - Prompt Store   │
│ + context   │ + change_req  │  - Evaluation     │
│ → prompt    │ + examples    │  - DSPy Config    │
│             │ → new_prompt  │                   │
└─────────────┴───────────────┴───────────────────┘
```

## DSPy Module Design

### Pipeline 1: CreatePrompt

**Signature:**
- Inputs: `description` (what the prompt should do), `context` (optional: audience, tone, constraints)
- Outputs: `prompt_text` (ready-to-use prompt), `reasoning` (why this structure was chosen)
- Module: `dspy.ChainOfThought` for step-by-step prompt construction

The module analyzes the description, identifies key behaviors, and structures them into a prompt following best practices (clear role, instructions, constraints, output format).

### Pipeline 2: IteratePrompt

**Signature:**
- Inputs: `current_prompt` (existing prompt), `change_request` (what to add/modify/fix), `failing_examples` (optional input/output pairs where current prompt fails)
- Outputs: `improved_prompt` (updated prompt), `changes_made` (summary of changes and rationale)
- Module: `dspy.ChainOfThought` for reasoning about what to preserve vs. modify

The module diffs the request against the current prompt, incorporates lessons from failing examples, and produces the improved version with a changelog.

### AI-as-Judge Metric

**PromptQualityJudge signature:**
- Inputs: `prompt_text`, `original_description`
- Outputs: `quality_score` (0.0–1.0), `feedback` (specific suggestions)

Used by both pipelines. For iteration, an additional comparison judge scores whether the new prompt improves on the old one.

### Example-Based Metric

When the user provides input/output evaluation pairs, a metric function runs the generated prompt against those examples and measures correctness. This gives stronger signal than AI-as-judge alone.

## Evaluation Strategy

- **No examples available:** AI-as-judge scores prompt quality (clarity, completeness, specificity)
- **Examples available:** Run the prompt against examples, measure output correctness
- **Both available:** Combine scores (weighted average)
- **Trace-aware strictness:** Strict during bootstrapping (both metrics must pass), lenient during evaluation (average score)

## File-Based Versioning

Prompts stored as JSON files in `prompts/{prompt_name}/v1.json`, `v2.json`, etc.

Each version contains:
- `version`, `parent_version` (lineage tracking)
- `prompt_text`, `description`, `change_request`, `changes_made`
- `quality_score`, `judge_feedback`
- `timestamp`, `metadata` (pipeline, model)

The iteration pipeline auto-increments by reading the latest version.

## Optional Optimizer Runner

Convenience utility for running DSPy optimizers:
- `BootstrapFewShot` — for < 20 examples, quick prototype
- `MIPROv2(auto="medium")` — for 200+ examples, joint instruction + demo tuning

Users can also call DSPy optimizers directly on the modules.

## Project Structure

```
DSPy_test/
├── src/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── create_prompt.py      # Pipeline 1 module
│   │   └── iterate_prompt.py     # Pipeline 2 module
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── judge.py              # AI-as-judge metric
│   │   └── example_metric.py     # Example-based metric
│   ├── store/
│   │   ├── __init__.py
│   │   └── prompt_store.py       # File-based versioning
│   └── config.py                 # DSPy LM configuration
├── prompts/                      # Generated prompt versions
├── examples/
│   ├── create_example.py
│   └── iterate_example.py
├── tests/
│   ├── test_create_prompt.py
│   ├── test_iterate_prompt.py
│   └── test_prompt_store.py
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **Hybrid approach (zero-data + optimization):** Works immediately, improves with data
2. **Single prompts only:** Keeps scope focused; multi-step chaining can be added later
3. **File-based versioning:** No extra dependencies, git-friendly, easy to inspect
4. **OpenAI target:** Prompts optimized for GPT-4o family
5. **Developer-focused API:** Python-first, no CLI/UI layer
