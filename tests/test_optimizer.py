import pytest
from unittest.mock import patch, MagicMock

import dspy

from src.optimizer import OptimizerRunner


class TestSelectOptimizer:
    def test_bootstrap_for_small_data(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=10) is dspy.BootstrapFewShot

    def test_mipro_for_large_data(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=200) is dspy.MIPROv2

    def test_threshold_boundary_below(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=49) is dspy.BootstrapFewShot

    def test_threshold_boundary_at(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=50) is dspy.MIPROv2

    def test_explicit_override(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=10, optimizer_name="MIPROv2") is dspy.MIPROv2

    def test_explicit_override_bootstrap(self):
        runner = OptimizerRunner()
        assert runner.select_optimizer(num_examples=200, optimizer_name="BootstrapFewShot") is dspy.BootstrapFewShot

    def test_invalid_optimizer_name_raises(self):
        runner = OptimizerRunner()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            runner.select_optimizer(num_examples=10, optimizer_name="NonexistentOptimizer")

    def test_non_class_attribute_raises(self):
        runner = OptimizerRunner()
        with pytest.raises(ValueError, match="not an optimizer class"):
            runner.select_optimizer(num_examples=10, optimizer_name="configure")


class TestOptimize:
    """Test that optimize() correctly initializes each optimizer type with the right params.

    We mock the optimizer *classes* (not the whole dspy module) so that
    OptimizerRunner.select_optimizer still returns a real class that we control.
    """

    def test_bootstrap_gets_correct_init_params(self):
        mock_instance = MagicMock()
        mock_instance.compile.return_value = MagicMock(spec=dspy.Module)
        mock_cls = MagicMock(return_value=mock_instance, __name__="BootstrapFewShot")

        runner = OptimizerRunner()
        metric = lambda ex, pred, trace=None: True
        trainset = [MagicMock()] * 10

        with patch.object(runner, "select_optimizer", return_value=mock_cls):
            runner.optimize(program=MagicMock(spec=dspy.Module), trainset=trainset, metric=metric)

        mock_cls.assert_called_once_with(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=8)
        mock_instance.compile.assert_called_once()

    def test_mipro_gets_correct_init_params(self):
        mock_instance = MagicMock()
        mock_instance.compile.return_value = MagicMock(spec=dspy.Module)
        mock_cls = MagicMock(return_value=mock_instance, __name__="MIPROv2")

        runner = OptimizerRunner()
        metric = lambda ex, pred, trace=None: True
        trainset = [MagicMock()] * 100

        with patch.object(runner, "select_optimizer", return_value=mock_cls):
            runner.optimize(program=MagicMock(spec=dspy.Module), trainset=trainset, metric=metric)

        mock_cls.assert_called_once_with(metric=metric, auto="medium")

    def test_unknown_optimizer_gets_generic_init(self):
        mock_instance = MagicMock()
        mock_instance.compile.return_value = MagicMock(spec=dspy.Module)
        mock_cls = MagicMock(return_value=mock_instance, __name__="SomeOtherOptimizer")

        runner = OptimizerRunner()
        metric = lambda ex, pred, trace=None: True

        with patch.object(runner, "select_optimizer", return_value=mock_cls):
            runner.optimize(program=MagicMock(spec=dspy.Module), trainset=[MagicMock()], metric=metric)

        # Generic path: only metric, no extra params
        mock_cls.assert_called_once_with(metric=metric)

    def test_save_path_triggers_save(self, tmp_path):
        mock_instance = MagicMock()
        mock_optimized = MagicMock(spec=dspy.Module)
        mock_instance.compile.return_value = mock_optimized
        mock_cls = MagicMock(return_value=mock_instance, __name__="BootstrapFewShot")

        runner = OptimizerRunner()
        save_file = str(tmp_path / "optimized.json")

        with patch.object(runner, "select_optimizer", return_value=mock_cls):
            result = runner.optimize(
                program=MagicMock(spec=dspy.Module),
                trainset=[MagicMock()] * 10,
                metric=lambda ex, pred, trace=None: True,
                save_path=save_file,
            )

        mock_optimized.save.assert_called_once_with(save_file)
        assert result is mock_optimized

    def test_no_save_when_path_not_given(self):
        mock_instance = MagicMock()
        mock_optimized = MagicMock(spec=dspy.Module)
        mock_instance.compile.return_value = mock_optimized
        mock_cls = MagicMock(return_value=mock_instance, __name__="BootstrapFewShot")

        runner = OptimizerRunner()

        with patch.object(runner, "select_optimizer", return_value=mock_cls):
            result = runner.optimize(
                program=MagicMock(spec=dspy.Module),
                trainset=[MagicMock()] * 10,
                metric=lambda ex, pred, trace=None: True,
            )

        mock_optimized.save.assert_not_called()
        assert result is mock_optimized
