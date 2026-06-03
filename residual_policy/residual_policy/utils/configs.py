from dataclasses import dataclass, field
from typing import List


@dataclass
class ResidualPolicyConfig:
    """
    Config for residual policy fine-tuning.

    mode choices:
      "reward"    — reward-conditioned residual: the residual head takes
                    [z_obs, action_dp, reward_feat] as input. Requires a
                    pre-trained reward model (reward_model_ckpt).
      "advantage" — advantage-weighted BC: weights each sample by its
                    per-trajectory advantage (r_t - mean(r_τ)).
      "topk"      — top-k filtered BC: keeps only frames with reward above
                    the top_rate quantile, then trains with uniform weights.
    """
    mode: str = "reward"            # "reward" | "advantage" | "topk"
    experiment_name: str = ""

    # ── Data ──────────────────────────────────────────────────────────────
    dataset_path: List[str] = field(default_factory=list)
    memmap_dir: str = ""
    use_memmap: bool = True
    fruit: str = "apple"            # task name: "apple" | "potato"
    reward_alpha: float = 0.8       # blending weight for global vs local reward
    binary_reward: bool = False     # use binary (0/1) reward tables instead of graded

    # ── Diffusion policy ──────────────────────────────────────────────────
    dp_ckpt: str = ""               # path to the base diffusion policy checkpoint
    obs_dim: int = 192              # encoder output dimension
    act_dim: int = 6                # action dimension

    # ── Training ──────────────────────────────────────────────────────────
    device: str = "cuda:0"
    seed: int = 0
    batch_size: int = 256
    num_workers: int = 8            # DataLoader workers (reward mode only)
    lr: float = 1e-4
    epochs: int = 3
    lam: float = 0.5               # residual blending: a_final = a_dp + lam * delta
    beta: float = 2.0              # temperature for exponential weighting
    delta_norm_max: float = 2.0    # max L2 norm of the residual action
    delta_reg: float = 1e-3        # L2 penalty coefficient on the residual delta
    max_grad_norm: float = 1.0

    # ── Reward-conditioned mode ────────────────────────────────────────────
    reward_feat_dim: int = 16       # projected reward feature size
    reward_model_ckpt: str = ""     # required when mode="reward"

    # ── Top-K filtering mode ──────────────────────────────────────────────
    top_rate: float = 0.7           # keep frames with reward >= this quantile
                                    # (e.g. 0.7 = keep top 30%)

    # ── Logging / checkpointing ────────────────────────────────────────────
    entity: str = ""                # wandb entity (empty = use default)
    project: str = "residual-dp"
    run_name: str = "residual_policy"
    save_dir: str = "./checkpoints"
    log_every: int = 100
    watch_model: bool = False


@dataclass
class RewardModelConfig:
    """Config for training the reward model used by the reward-conditioned residual policy."""

    # ── Data ──────────────────────────────────────────────────────────────
    dataset_path: List[str] = field(default_factory=list)
    use_memmap: bool = True
    memmap_dir: str = ""
    fruit: str = "apple"            # task name: "apple" | "potato"
    reward_alpha: float = 0.8       # blending weight for global vs local reward
    binary_reward: bool = False     # use binary (0/1) reward tables instead of graded

    # ── Diffusion encoder ─────────────────────────────────────────────────
    dp_ckpt: str = ""               # path to the base diffusion policy checkpoint
    obs_dim: int = 192              # encoder output dimension
    act_dim: int = 6                # action dimension
    hidden_dim: int = 256
    num_layers: int = 2

    # ── Training ──────────────────────────────────────────────────────────
    device: str = "cuda:0"
    seed: int = 0
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    valid_ratio: float = 0.1
    norm_target: bool = True        # z-score normalise the regression target
    num_workers: int = 8
    max_grad_norm: float = 1.0

    # ── Logging / checkpointing ────────────────────────────────────────────
    entity: str = ""                # wandb entity (empty = use default)
    project: str = "reward-model"
    run_name: str = "rm_apple"
    save_dir: str = "./reward_model_ckpts"
    log_every: int = 100
    eval_every: int = 1000
    watch_model: bool = False
