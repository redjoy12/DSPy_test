"""PromptForge — Streamlit UI for DSPy-powered prompt engineering."""

import difflib
import json
import os
import streamlit as st

from src.config import configure_lm
from src.store.prompt_store import PromptStore, PromptVersion


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "api_key": "",
        "models": [],
        "selected_model": None,
        "prompts_dir": "prompts",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_models(api_key: str) -> list[str]:
    """Call OpenAI API and return sorted list of chat model IDs."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    models = client.models.list()
    chat_prefixes = ("gpt-3.5", "gpt-4", "chatgpt", "o1", "o3", "o4")
    chat_models = sorted(
        m.id for m in models.data if m.id.startswith(chat_prefixes)
    )
    return chat_models


def format_model_for_dspy(model_id: str) -> str:
    """Prepend ``openai/`` prefix if missing (DSPy convention)."""
    if not model_id.startswith("openai/"):
        return f"openai/{model_id}"
    return model_id


def validate_prompt_name(name: str) -> str:
    """Reject names with path separators or traversal components."""
    name = name.strip()
    PromptStore.validate_name(name)
    return name


def validate_relative_path(path: str) -> str:
    """Reject absolute paths and '..' components."""
    from pathlib import PurePath
    p = PurePath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Path must be relative and must not contain '..' components.")
    return path


def get_store() -> PromptStore:
    return PromptStore(base_dir=st.session_state["prompts_dir"])


def ensure_lm_configured(model: str):
    """Set the API key env-var and configure DSPy's global LM.

    Note: The API key is set as a process-level environment variable. In
    multi-user Streamlit deployments (Community Cloud, shared servers), all
    users share one Python process, so one user's key overwrites another's.
    This is intended for local single-user use only.
    """
    os.environ["OPENAI_API_KEY"] = st.session_state["api_key"]
    configure_lm(model=format_model_for_dspy(model))


def require_api_key() -> bool:
    """Return True if an API key is set; show warning otherwise."""
    if not st.session_state["api_key"]:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return False
    return True


def display_score(label: str, score: float):
    """Render a score with color coding."""
    if score >= 0.8:
        color = "green"
    elif score >= 0.5:
        color = "yellow"
    else:
        color = "red"
    st.markdown(f"**{label}:** :{color}[{score:.2f}]")


def render_diff(text_a: str, text_b: str, label_a: str = "Before", label_b: str = "After"):
    """Show a unified diff between two texts."""
    diff = difflib.unified_diff(
        text_a.splitlines(keepends=True),
        text_b.splitlines(keepends=True),
        fromfile=label_a,
        tofile=label_b,
    )
    diff_text = "".join(diff)
    if diff_text:
        st.code(diff_text, language="diff")
    else:
        st.info("No differences found.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Settings")

        # API key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state["api_key"],
            help="Required for Create, Iterate, and Optimize tabs.",
        )
        if st.button("Connect", disabled=not api_key):
            st.session_state["api_key"] = api_key
            try:
                with st.spinner("Fetching models..."):
                    st.session_state["models"] = fetch_models(api_key)
            except Exception as exc:
                st.error(f"Failed to fetch models: {exc}")
                st.session_state["models"] = []
        if not api_key and st.session_state["api_key"]:
            st.session_state["api_key"] = ""
            st.session_state["models"] = []

        # Status indicator
        if st.session_state["api_key"]:
            st.success("API key configured")
        else:
            st.info("API key not set")

        # Multi-user deployment warning
        _shared_env_vars = (
            "STREAMLIT_SHARING_MODE",
            "STREAMLIT_SERVER_ADDRESS",
            "IS_COMMUNITY_CLOUD",
        )
        if any(os.environ.get(v) for v in _shared_env_vars):
            st.warning(
                "Multi-user deployment detected. API keys are stored "
                "process-wide and may leak between users."
            )

        # Model selector
        models = st.session_state["models"]
        if models:
            prev = st.session_state.get("selected_model")
            default_idx = models.index(prev) if prev in models else 0
            selected = st.selectbox("Default model", models, index=default_idx)
            st.session_state["selected_model"] = selected
        else:
            st.selectbox("Default model", ["(enter API key first)"], disabled=True)

        # Prompts directory
        prompts_dir = st.text_input(
            "Prompts directory",
            value=st.session_state["prompts_dir"],
            help="Directory where prompt versions are stored (relative path only).",
        )
        from pathlib import PurePath
        p = PurePath(prompts_dir)
        if p.is_absolute() or ".." in p.parts:
            st.error("Prompts directory must be a relative path without '..' components.")
        else:
            if prompts_dir != st.session_state["prompts_dir"]:
                st.session_state["prompts_dir"] = prompts_dir
                st.rerun()


# ---------------------------------------------------------------------------
# Tab 3: Browse Versions
# ---------------------------------------------------------------------------

def render_browse_tab():
    store = get_store()
    prompts = store.list_prompts()

    if not prompts:
        st.info("No prompts found. Create one in the **Create** tab first.")
        return

    selected_name = st.selectbox("Prompt name", prompts, key="browse_prompt_name")

    versions = store.list_versions(selected_name)
    if not versions:
        st.info("No versions found for this prompt.")
        return

    # --- Version detail ---
    st.subheader("Version Detail")
    selected_version = st.selectbox(
        "Version", versions, index=len(versions) - 1, key="browse_version"
    )
    ver = store.load(selected_name, selected_version)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Version:** {ver.version}")
        st.markdown(f"**Pipeline:** {ver.pipeline}")
        st.markdown(f"**Model:** {ver.model}")
        display_score("Quality score", ver.quality_score)
    with col2:
        st.markdown(f"**Timestamp:** {ver.timestamp}")
        st.markdown(f"**Parent version:** {ver.parent_version or 'None'}")
        if ver.description:
            st.markdown(f"**Description:** {ver.description}")

    st.text_area("Prompt text", ver.prompt_text, height=200, disabled=True, key="browse_prompt_text")
    st.text_area("Judge feedback", ver.judge_feedback, height=100, disabled=True, key="browse_feedback")

    if ver.change_request:
        st.markdown(f"**Change request:** {ver.change_request}")
    if ver.changes_made:
        st.markdown(f"**Changes made:** {ver.changes_made}")

    # --- Side-by-side comparison ---
    if len(versions) >= 2:
        st.divider()
        st.subheader("Compare Versions")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            v_a = st.selectbox("Version A", versions, index=0, key="compare_a")
        with comp_col2:
            v_b = st.selectbox(
                "Version B", versions, index=len(versions) - 1, key="compare_b"
            )

        if v_a != v_b:
            ver_a = ver if v_a == selected_version else store.load(selected_name, v_a)
            ver_b = ver if v_b == selected_version else store.load(selected_name, v_b)
            render_diff(ver_a.prompt_text, ver_b.prompt_text, f"v{v_a}", f"v{v_b}")
        else:
            st.info("Select two different versions to compare.")


# ---------------------------------------------------------------------------
# Tab 1: Create Prompt
# ---------------------------------------------------------------------------

def render_create_tab():
    if not require_api_key():
        return

    models = st.session_state["models"]
    if not models:
        st.warning("No models available. Check your API key.")
        return

    with st.form("create_form"):
        name = st.text_input("Prompt name", max_chars=100, help="Identifier for this prompt (e.g. 'email-assistant').")
        description = st.text_input("Description", help="What the prompt should do.")
        context = st.text_area(
            "Context (optional)", help="Target audience, tone, constraints, etc.",
            max_chars=5000,
        )
        model = st.selectbox("Model", models, key="create_model")
        min_score = st.slider(
            "Minimum quality score", 0.0, 1.0, 0.0, 0.05,
            help="Set above 0 to reject low-quality prompts.",
        )
        submitted = st.form_submit_button("Create Prompt")

    if submitted:
        if not name or not description:
            st.error("Name and description are required.")
            return

        try:
            name = validate_prompt_name(name)
            ensure_lm_configured(model)
            from src.pipelines.create_prompt import CreatePromptPipeline

            store = get_store()
            pipeline = CreatePromptPipeline(store=store)
            with st.spinner("Generating prompt..."):
                version = pipeline.create_and_save(
                    name=name,
                    description=description,
                    context=context,
                    min_score=min_score if min_score > 0 else None,
                )

            st.session_state["create_result"] = {
                "name": name,
                "version": version.version,
                "prompt_text": version.prompt_text,
                "quality_score": version.quality_score,
                "judge_feedback": version.judge_feedback,
            }

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Error: {exc}")

    # Display results from session state (survives reruns)
    result = st.session_state.get("create_result")
    if result:
        st.success(f"Prompt **{result['name']}** v{result['version']} created!")
        st.text_area("Generated prompt", result["prompt_text"], height=200, disabled=True, key="create_prompt_text")
        display_score("Quality score", result["quality_score"])
        st.text_area("Judge feedback", result["judge_feedback"], height=100, disabled=True, key="create_feedback")
        if st.button("Clear results", key="clear_create"):
            del st.session_state["create_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Iterate Prompt
# ---------------------------------------------------------------------------

def render_iterate_tab():
    if not require_api_key():
        return

    models = st.session_state["models"]
    if not models:
        st.warning("No models available. Check your API key.")
        return

    store = get_store()
    prompts = store.list_prompts()

    if not prompts:
        st.info("No prompts found. Create one in the **Create** tab first.")
        return

    selected_name = st.selectbox("Prompt name", prompts, key="iterate_prompt_name")

    # Show current latest prompt
    try:
        latest = store.load_latest(selected_name)
        st.text_area(
            f"Current prompt (v{latest.version})",
            latest.prompt_text,
            height=150,
            disabled=True,
            key="iterate_current",
        )
        before_text = latest.prompt_text
    except FileNotFoundError:
        st.warning("Could not load latest version.")
        return

    with st.form("iterate_form"):
        change_request = st.text_area("Change request", help="What to add, modify, or fix.", max_chars=5000)
        failing_examples = st.text_area(
            "Failing examples (optional)",
            help="Input/output pairs where the current prompt fails.",
            max_chars=10000,
        )
        model = st.selectbox("Model", models, key="iterate_model")
        min_score = st.slider(
            "Minimum improvement score", 0.0, 1.0, 0.0, 0.05,
            help="Set above 0 to reject low-quality iterations.",
        )
        submitted = st.form_submit_button("Iterate Prompt")

    if submitted:
        if not change_request:
            st.error("Change request is required.")
            return

        try:
            ensure_lm_configured(model)
            from src.pipelines.iterate_prompt import IteratePromptPipeline

            pipeline = IteratePromptPipeline(store=store)
            with st.spinner("Iterating prompt..."):
                version = pipeline.iterate_and_save(
                    name=selected_name,
                    change_request=change_request,
                    current_prompt=before_text,
                    description=latest.description,
                    failing_examples=failing_examples,
                    min_score=min_score if min_score > 0 else None,
                )

            st.session_state["iterate_result"] = {
                "name": selected_name,
                "before_text": before_text,
                "parent_version": version.parent_version,
                "version": version.version,
                "prompt_text": version.prompt_text,
                "quality_score": version.quality_score,
                "changes_made": version.changes_made,
                "judge_feedback": version.judge_feedback,
            }

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Error: {exc}")

    # Display results from session state (survives reruns)
    result = st.session_state.get("iterate_result")
    if result and result["name"] == selected_name:
        st.success(f"Prompt **{result['name']}** v{result['version']} created!")

        saved_before = result.get("before_text", "")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before**")
            st.text_area("Before", saved_before, height=200, disabled=True, key="iter_before")
        with col2:
            st.markdown("**After**")
            st.text_area("After", result["prompt_text"], height=200, disabled=True, key="iter_after")

        display_score("Improvement score", result["quality_score"])
        if result["changes_made"]:
            st.markdown(f"**Changes made:** {result['changes_made']}")
        st.text_area("Judge feedback", result["judge_feedback"], height=100, disabled=True, key="iter_feedback")
        if st.button("Clear results", key="clear_iterate"):
            del st.session_state["iterate_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: Optimize
# ---------------------------------------------------------------------------

def render_optimize_tab():
    if not require_api_key():
        return

    models = st.session_state["models"]
    if not models:
        st.warning("No models available. Check your API key.")
        return

    with st.form("optimize_form"):
        pipeline_choice = st.selectbox(
            "Pipeline", ["Create", "Iterate"], key="optimize_pipeline"
        )

        training_json = st.text_area(
            "Training examples (JSON)",
            height=200,
            max_chars=50000,
            help=(
                "JSON array of objects. For Create: [{\"description\": ..., \"context\": ...}]. "
                "For Iterate: [{\"current_prompt\": ..., \"change_request\": ...}]."
            ),
        )

        save_path = st.text_input(
            "Save path (optional)",
            help="File path to save the optimized program (use a .json extension).",
        )

        model = st.selectbox("Model", models, key="optimize_model")

        st.caption(
            "Cost estimate: BootstrapFewShot makes ~1× LLM calls per example; "
            "MIPROv2 makes ~10× or more. Larger training sets cost proportionally more."
        )
        cost_confirmed = st.checkbox(
            "I understand this may trigger many LLM calls and incur API costs."
        )

        submitted = st.form_submit_button("Run Optimization")

    if submitted:
        if not cost_confirmed:
            st.error("Please confirm you understand the API cost before running optimization.")
            return

        if not training_json.strip():
            st.error("Training examples are required.")
            return

        if save_path:
            try:
                validate_relative_path(save_path)
            except ValueError as exc:
                st.error(str(exc))
                return
            from pathlib import Path
            if not save_path.endswith(".json"):
                st.warning("Save path should end with '.json' for DSPy compatibility.")
            parent = Path(save_path).parent
            if str(parent) != '.' and not parent.exists():
                st.error(f"Directory '{parent}' does not exist.")
                return

        try:
            parsed = json.loads(training_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return

        if not isinstance(parsed, list) or not parsed:
            st.error("Training examples must be a non-empty JSON array.")
            return

        try:
            import dspy
            from src.evaluation.judge import make_comparison_metric, make_quality_metric
            from src.optimizer import OptimizerRunner

            ensure_lm_configured(model)

            if pipeline_choice == "Create":
                input_keys = ["description", "context"]
            else:
                input_keys = ["current_prompt", "change_request", "failing_examples"]
            for i, ex in enumerate(parsed):
                missing = [k for k in input_keys if k not in ex]
                if missing:
                    st.error(f"Example {i + 1} is missing keys: {', '.join(missing)}")
                    return
                extra = [k for k in ex if k not in input_keys]
                if extra:
                    st.warning(f"Example {i + 1} has extra keys that will be passed through: {', '.join(extra)}")
            trainset = [dspy.Example(**ex).with_inputs(*input_keys) for ex in parsed]

            runner = OptimizerRunner()
            optimizer_cls = runner.select_optimizer(len(parsed))
            st.info(f"Using optimizer: **{optimizer_cls.__name__}** (based on {len(parsed)} examples)")

            if pipeline_choice == "Create":
                from src.pipelines.create_prompt import CreatePromptPipeline
                program = CreatePromptPipeline(store=get_store())
                metric = make_quality_metric()
            else:
                from src.pipelines.iterate_prompt import IteratePromptPipeline
                program = IteratePromptPipeline(store=get_store())
                metric = make_comparison_metric()

            with st.spinner("Running optimization (this may take a while)..."):
                optimized = runner.optimize(
                    program=program,
                    trainset=trainset,
                    metric=metric,
                    save_path=save_path if save_path else None,
                )

            demos_by_predictor = {}
            for pred_name, param in optimized.named_predictors():
                demos = getattr(param, "demos", [])
                if demos:
                    demos_by_predictor[pred_name] = [
                        {k: str(v) for k, v in demo.items()} for demo in demos
                    ]
            num_demos = sum(len(d) for d in demos_by_predictor.values())

            st.session_state["optimize_result"] = {
                "num_demos": num_demos,
                "demos_by_predictor": demos_by_predictor,
                "save_path": save_path if save_path else None,
            }

        except Exception as exc:
            st.error(f"Optimization failed: {exc}")

    # Display results from session state (survives reruns)
    result = st.session_state.get("optimize_result")
    if result:
        st.success("Optimization complete!")
        num_demos = result["num_demos"]
        if num_demos == 0:
            st.warning(
                "No bootstrapped demos were produced. The optimizer may not "
                "have found examples that passed the metric. Consider adding "
                "more diverse training examples."
            )
        else:
            st.info(f"Optimization produced **{num_demos}** bootstrapped demo(s) across all predictors.")
            for pred_name, demos in result["demos_by_predictor"].items():
                with st.expander(f"Bootstrapped demos for `{pred_name}` ({len(demos)})"):
                    for i, demo in enumerate(demos):
                        st.markdown(f"**Demo {i + 1}**")
                        st.json(demo)
        if result["save_path"]:
            st.info(f"Optimized program saved to `{result['save_path']}`.")
        if st.button("Clear results", key="clear_optimize"):
            del st.session_state["optimize_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 5: Upload Jinja2 Template
# ---------------------------------------------------------------------------

def render_upload_tab():
    uploaded = st.file_uploader(
        "Upload a Jinja2 prompt template",
        type=["jinja2", "j2"],
        help="Upload a .jinja2 or .j2 file. The raw template text will be stored as the prompt.",
    )

    if uploaded is not None:
        try:
            template_text = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("File is not valid UTF-8 text.")
            return

        st.text_area(
            "Template preview",
            template_text,
            height=250,
            disabled=True,
            key="upload_preview",
        )

        with st.form("upload_form"):
            name = st.text_input(
                "Prompt name",
                max_chars=100,
                help="Identifier for this prompt (e.g. 'email-assistant').",
            )
            description = st.text_input(
                "Description",
                help="What this prompt template does.",
            )
            submitted = st.form_submit_button("Save as Prompt")

        if submitted:
            if not name or not description:
                st.error("Name and description are required.")
                return

            try:
                name = validate_prompt_name(name)
                store = get_store()
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

                st.session_state["upload_result"] = {
                    "name": name,
                    "version": version_num,
                }
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Error: {exc}")

    result = st.session_state.get("upload_result")
    if result:
        st.success(f"Saved **{result['name']}** v{result['version']}. You can now iterate on it in the **Iterate** tab.")
        if st.button("Clear results", key="clear_upload"):
            del st.session_state["upload_result"]
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PromptForge", layout="wide")
    st.title("PromptForge")
    st.caption("DSPy-powered prompt engineering")

    init_session_state()
    render_sidebar()

    tab_create, tab_iterate, tab_browse, tab_optimize, tab_upload = st.tabs(
        ["Create", "Iterate", "Browse", "Optimize", "Upload"]
    )

    with tab_create:
        render_create_tab()
    with tab_iterate:
        render_iterate_tab()
    with tab_browse:
        render_browse_tab()
    with tab_optimize:
        render_optimize_tab()
    with tab_upload:
        render_upload_tab()


if __name__ == "__main__":
    main()
