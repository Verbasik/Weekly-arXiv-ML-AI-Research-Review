from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrainingConfig:
    """
    Phase 4 training hyperparameters (skeleton).

    TODO:
    - Tune batch_size and gradient_accumulation_steps for A100
    - Decide on mixed_precision: None / 'bf16' / 'fp16'
    - Adjust eval_every/save_every intervals
    """

    # Data
    dataset_name: str = "wikitext-2"
    seq_len: int = 2048  # should match Qwen3Config.max_position_embeddings
    num_workers: int = 2

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 2000
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 100
    alpha_balance: float = 0.01

    # Precision / Device
    mixed_precision: Optional[str] = None
    device: str = "cuda"

    # Logging / Eval / Checkpoints
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 1000
    output_dir: str = "checkpoints/qwen3_moe_wikitext2"
    max_eval_batches: int = 100
    seed: int = 42
