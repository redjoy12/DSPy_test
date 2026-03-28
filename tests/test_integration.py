"""
Integration tests for the full create -> iterate -> evaluate round-trip.
Uses DummyLM so the real DSPy pipeline runs without API calls.
"""
import pytest
import dspy
from dspy.utils import DummyLM

from src.pipelines.create_prompt import CreatePromptPipeline
from src.pipelines.iterate_prompt import IteratePromptPipeline
from src.evaluation.example_metric import ExampleBasedMetric
from src.store.prompt_store import PromptStore


class TestCreateThenIterate:
    def test_full_lineage(self, store):
        """Create v1, iterate to v2, verify the full version chain on disk."""
        # Step 1: Create
        create_lm = DummyLM([
            {"reasoning": "Built for coding help.", "prompt_text": "You are a helpful coding assistant."},
            {"reasoning": "Solid start.", "quality_score": "0.85", "feedback": "Clear role definition."},
        ])
        with dspy.context(lm=create_lm):
            create_pipeline = CreatePromptPipeline(store=store)
            v1 = create_pipeline.create_and_save(
                name="coding_assistant",
                description="A coding assistant for Python developers",
            )

        assert v1.version == 1
        assert v1.parent_version is None
        assert v1.pipeline == "create"

        # Step 2: Iterate
        iterate_lm = DummyLM([
            {
                "reasoning": "Added error handling behavior.",
                "improved_prompt": "You are a helpful coding assistant. When errors occur, explain them clearly.",
                "changes_made": "Added error explanation behavior.",
            },
            {"reasoning": "Good addition.", "improvement_score": "0.91", "feedback": "Addresses the request."},
        ])
        with dspy.context(lm=iterate_lm):
            iterate_pipeline = IteratePromptPipeline(store=store)
            v2 = iterate_pipeline.iterate_and_save(
                name="coding_assistant",
                change_request="Add error explanation behavior",
            )

        assert v2.version == 2
        assert v2.parent_version == 1
        assert v2.pipeline == "iterate"
        assert v2.change_request == "Add error explanation behavior"

        # Step 3: Verify lineage on disk
        all_versions = store.list_versions("coding_assistant")
        assert all_versions == [1, 2]

        loaded_v1 = store.load("coding_assistant", 1)
        loaded_v2 = store.load("coding_assistant", 2)
        assert loaded_v2.parent_version == loaded_v1.version
        assert loaded_v1.prompt_text != loaded_v2.prompt_text

    def test_multiple_iterations_build_chain(self, store):
        """Four versions created sequentially form a proper chain."""
        # Create v1
        lm = DummyLM([
            {"reasoning": "Initial.", "prompt_text": "V1 prompt."},
            {"reasoning": "OK.", "quality_score": "0.8", "feedback": "Baseline."},
        ])
        with dspy.context(lm=lm):
            CreatePromptPipeline(store=store).create_and_save(name="evolving", description="Test")

        # Iterate v2, v3, v4
        for i in range(2, 5):
            lm = DummyLM([
                {"reasoning": f"Change {i}.", "improved_prompt": f"V{i} prompt.", "changes_made": f"Change {i}."},
                {"reasoning": "x.", "improvement_score": "0.9", "feedback": "Improved."},
            ])
            with dspy.context(lm=lm):
                IteratePromptPipeline(store=store).iterate_and_save(
                    name="evolving", change_request=f"Improvement {i}",
                )

        versions = store.list_versions("evolving")
        assert versions == [1, 2, 3, 4]

        # Verify chain integrity
        for v in range(2, 5):
            loaded = store.load("evolving", v)
            assert loaded.parent_version == v - 1


class TestEndToEndWithMetric:
    def test_created_prompt_can_be_evaluated(self, store):
        """A prompt created by the pipeline can be evaluated with ExampleBasedMetric."""
        # Create a prompt
        create_lm = DummyLM([
            {"reasoning": "Greeting bot.", "prompt_text": "You are a friendly greeter."},
            {"reasoning": "Good.", "quality_score": "0.9", "feedback": "Nice."},
        ])
        with dspy.context(lm=create_lm):
            pipeline = CreatePromptPipeline(store=store)
            v1 = pipeline.create_and_save(name="greeter", description="A greeting bot")

        # Evaluate the stored prompt against examples
        eval_lm = DummyLM([
            {"output": "Hello! Welcome!"},
            {"output": "Hi there, friend!"},
        ])
        examples = [
            {"input": "Hi", "expected_output": "Hello! Welcome!"},
            {"input": "Hey", "expected_output": "Hi there, friend!"},
        ]
        with dspy.context(lm=eval_lm):
            metric = ExampleBasedMetric()
            loaded = store.load("greeter", 1)
            score = metric.evaluate(loaded.prompt_text, examples)

        assert score == 1.0


class TestMultiplePromptsIsolation:
    def test_separate_prompts_dont_interfere(self, store):
        """Two different prompt names maintain independent version histories."""
        lm = DummyLM([
            {"reasoning": "Alpha.", "prompt_text": "Alpha prompt."},
            {"reasoning": "OK.", "quality_score": "0.8", "feedback": "Fine."},
            {"reasoning": "Beta.", "prompt_text": "Beta prompt."},
            {"reasoning": "OK.", "quality_score": "0.85", "feedback": "Good."},
        ])
        with dspy.context(lm=lm):
            pipeline = CreatePromptPipeline(store=store)
            alpha = pipeline.create_and_save(name="alpha", description="Alpha prompt")
            beta = pipeline.create_and_save(name="beta", description="Beta prompt")

        assert store.list_prompts() == ["alpha", "beta"]
        assert store.list_versions("alpha") == [1]
        assert store.list_versions("beta") == [1]
        assert store.load("alpha", 1).prompt_text != store.load("beta", 1).prompt_text
