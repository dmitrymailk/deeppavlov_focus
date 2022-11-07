from dataclasses import dataclass
from typing import Optional


@dataclass
class LightingHyperparametersV1:
    precision: int | str = 32
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    auto_scale_batch_size: Optional[str] = None
    # profiler: str = "simple"
    deterministic: bool = True
    max_epochs: int = 1
