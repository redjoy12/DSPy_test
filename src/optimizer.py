from __future__ import annotations
from typing import Callable, Optional

import dspy


class OptimizerRunner:
    OPTIMIZER_THRESHOLD = 50  # Below this, use BootstrapFewShot; above, use MIPROv2

    def select_optimizer(
        self,
        num_examples: int,
        optimizer_name: Optional[str] = None,
    ) -> type:
        if optimizer_name:
            try:
                cls = getattr(dspy, optimizer_name)
            except AttributeError:
                raise ValueError(
                    f"Unknown optimizer '{optimizer_name}'. "
                    f"Must be a valid dspy optimizer class (e.g. 'BootstrapFewShot', 'MIPROv2')."
                )
            if not isinstance(cls, type):
                raise ValueError(f"'{optimizer_name}' is not an optimizer class.")
            return cls
        if num_examples < self.OPTIMIZER_THRESHOLD:
            return dspy.BootstrapFewShot
        return dspy.MIPROv2

    def optimize(
        self,
        program: dspy.Module,
        trainset: list,
        metric: Callable,
        optimizer_name: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> dspy.Module:
        optimizer_cls = self.select_optimizer(
            num_examples=len(trainset),
            optimizer_name=optimizer_name,
        )

        cls_name = getattr(optimizer_cls, "__name__", "")
        if cls_name == "MIPROv2":
            optimizer = optimizer_cls(metric=metric, auto="medium", **kwargs)
        elif cls_name == "BootstrapFewShot":
            optimizer = optimizer_cls(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
                **kwargs,
            )
        else:
            optimizer = optimizer_cls(metric=metric, **kwargs)

        optimized = optimizer.compile(program, trainset=trainset)

        if save_path:
            optimized.save(save_path)

        return optimized
