"""Tests for the Jinja2 upload flow — verifies round-trip through PromptStore."""

from src.store.prompt_store import PromptStore, PromptVersion


SAMPLE_TEMPLATE = """\
You are a {{ role }} assistant.

{% if context %}
Context: {{ context }}
{% endif %}

Please help the user with: {{ task }}
"""


def _save_uploaded(store: PromptStore, name: str, template_text: str, description: str) -> PromptVersion:
    """Mimic what render_upload_tab does when the user clicks Save."""
    version_num = store.get_next_version(name)
    version = PromptVersion(
        version=version_num,
        parent_version=version_num - 1 if version_num > 1 else None,
        prompt_text=template_text,
        description=description,
        quality_score=0.0,
        judge_feedback="",
        pipeline="upload",
        model="n/a",
    )
    store.save(name, version)
    return version


def test_upload_saves_and_loads(store):
    ver = _save_uploaded(store, "my-template", SAMPLE_TEMPLATE, "A test template")

    assert ver.version == 1
    assert ver.parent_version is None
    assert ver.pipeline == "upload"
    assert ver.model == "n/a"
    assert ver.quality_score == 0.0

    loaded = store.load("my-template", 1)
    assert loaded.prompt_text == SAMPLE_TEMPLATE
    assert loaded.description == "A test template"
    assert loaded.pipeline == "upload"


def test_upload_appends_version(store):
    _save_uploaded(store, "my-template", "v1 content", "First upload")
    ver2 = _save_uploaded(store, "my-template", "v2 content", "Second upload")

    assert ver2.version == 2
    assert ver2.parent_version == 1

    latest = store.load_latest("my-template")
    assert latest.version == 2
    assert latest.prompt_text == "v2 content"


def test_upload_appears_in_list(store):
    _save_uploaded(store, "uploaded-prompt", SAMPLE_TEMPLATE, "desc")

    assert "uploaded-prompt" in store.list_prompts()
    assert store.list_versions("uploaded-prompt") == [1]
