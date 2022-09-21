from typing import Optional


class LightingHyperparametersV1:
    def __init__(
        self,
        precision: int | str = 32,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 0.0,
        auto_scale_batch_size: Optional[str] = None,
        profiler: str = "simple",
        deterministic: bool = True,
    ) -> None:
        self.precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.auto_scale_batch_size = auto_scale_batch_size
        self.profiler = profiler
        self.deterministic = deterministic
