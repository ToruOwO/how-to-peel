"""
Residual policy fine-tuning. Three training modes are available
(selected via config.residual.mode):

  reward     — reward-conditioned residual; uses a pre-trained RewardModel
               whose features are projected and concatenated to the obs
               embedding before being fed to the residual head.
  advantage  — advantage-weighted behavioural cloning; weights are derived
               from per-trajectory advantage A_t = r_t - mean(r_τ).
  topk       — top-K filtered BC; dataset is filtered to frames whose reward
               exceeds the top_rate quantile, then trained with uniform weights.
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from residual_policy.utils.configs import ResidualPolicyConfig
from residual_policy.common.models import ResidualHead, RewardModel, RewardFeatProjector
from residual_policy.dataset import (
    build_arrays_from_roots,
    build_dataset_from_memmap,
    ResidualPolicyDataloader,
    RewardDataset,
    collate_fn,
)
from diffusion_policy.export import build_diffusion_model_from_ckpt # This API should be by your diffusion policy implementation; adjust the import as needed.


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

MAX_WEIGHT = 20.0   # cap on per-sample importance weights (and clip-frac threshold)


def clip_by_norm(x: torch.Tensor, max_norm: float, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    n = torch.norm(x, p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x * (max_norm / n).clamp(max=1.0)


@torch.no_grad()
def dp_sample_with_z(dp_model, z_cond: torch.Tensor, device: str) -> torch.Tensor:
    """Run one DDIM pass through the diffusion policy given a conditioning embedding."""
    model, sched = dp_model.model, dp_model.scheduler
    B = z_cond.size(0)
    sample = torch.randn(B, dp_model.sample_dim, device=device)
    for t in sched.timesteps:
        t_tensor = torch.full((B,), int(t), dtype=torch.long, device=device)
        z_t = model.time_embedding(t_tensor)
        pred = model.denoiser(torch.cat([sample, z_cond, z_t], dim=-1))
        sample = sched.step(pred, t, sample).prev_sample
    return sample


def setup_wandb(cfg: ResidualPolicyConfig) -> Path:
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_name = f"{cfg.run_name}-{cfg.experiment_name}-{timestamp}"
    wandb.init(
        entity=cfg.entity or None,
        project=cfg.project,
        name=run_name,
        config=asdict(cfg),
        reinit=True,
        save_code=True,
    )
    wandb.define_metric("trainer/step")
    for prefix in ["loss/*", "train/*", "adv/*", "w/*", "delta/*"]:
        wandb.define_metric(prefix, step_metric="trainer/step")
    out_dir = Path(cfg.save_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_checkpoint(
    path: Path,
    res_model: nn.Module,
    meta: Dict[str, Any],
    reward_projector: Optional[nn.Module] = None,
) -> None:
    to_save = {"residual_state_dict": res_model.state_dict(), "meta": meta}
    if reward_projector is not None:
        to_save["reward_projector_state_dict"] = reward_projector.state_dict()
    torch.save(to_save, path)


def build_reward_model_from_ckpt(ckpt_path: str, obs_dim: int, act_dim: int, device: str) -> RewardModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("reward_model_state_dict", ckpt))
    model = RewardModel(obs_dim=obs_dim, act_dim=act_dim)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ──────────────────────────────────────────────────────────────────────────────
# Weight helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_weights_from_reward(
    reward: torch.Tensor,
    mean: float,
    std: float,
    beta: float = 2.0,
    max_w: float = MAX_WEIGHT,
) -> torch.Tensor:
    """Compute exponential importance weights from z-score-normalised rewards."""
    r = torch.nan_to_num(reward.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean_t = torch.as_tensor(mean, device=r.device, dtype=r.dtype)
    std_t  = torch.as_tensor(std,  device=r.device, dtype=r.dtype).clamp(min=1e-6)
    r = ((r - mean_t) / std_t).clamp(-5.0, 5.0)
    w = torch.exp(beta * r)
    w = torch.nan_to_num(w, nan=0.0, posinf=max_w, neginf=0.0).clamp(max=max_w)
    return w / (w.mean() + 1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Advantage helpers  (advantage mode only)
# ──────────────────────────────────────────────────────────────────────────────

def _build_traj_ids(terminals: np.ndarray) -> np.ndarray:
    t = np.asarray(terminals, dtype=np.float32).reshape(-1)
    ids = np.zeros(len(t), dtype=np.int64)
    cur = 0
    for i in range(len(t)):
        ids[i] = cur
        if t[i] > 0.5:
            cur += 1
    return ids


def _compute_advantage_weights(
    rewards: np.ndarray,
    traj_ids: np.ndarray,
    beta: float = 2.0,
    max_w: float = MAX_WEIGHT,
):
    r = np.nan_to_num(np.asarray(rewards, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    ids = np.asarray(traj_ids, dtype=np.int64).reshape(-1)
    A = np.zeros_like(r)
    for tid in np.unique(ids):
        mask = ids == tid
        A[mask] = r[mask] - r[mask].mean()
    np.clip(A, -5.0, 5.0, out=A)
    w = np.exp(beta * A).astype(np.float32)
    w = np.minimum(np.nan_to_num(w, nan=0.0, posinf=max_w, neginf=0.0), max_w)
    return A, w / (w.mean() + 1e-6)


def _add_advantage_weights(subset: dict, beta: float, max_w: float = MAX_WEIGHT) -> dict:
    r_np = np.asarray(subset["rewards"], dtype=np.float32).reshape(-1)
    t_np = np.asarray(subset["terminals"], dtype=np.float32).reshape(-1)
    A, w = _compute_advantage_weights(r_np, _build_traj_ids(t_np), beta=beta, max_w=max_w)
    print(f"  advantage  mean={A.mean():.4f}  std={A.std():.4f}")
    print(f"  weight     mean={w.mean():.4f}  std={w.std():.4f}  max={w.max():.4f}")
    subset["advantages"] = A.astype(np.float32)
    subset["weights"]    = w.astype(np.float32)
    return subset


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    train_loader,
    dp_model,
    cfg: ResidualPolicyConfig,
    run_meta: Dict[str, Any],
    out_dir: Path,
    reward_model: RewardModel,
):
    device = cfg.device

    # ── build model ──────────────────────────────────────────────────────────
    in_dim = cfg.obs_dim + cfg.act_dim + cfg.reward_feat_dim
    res_model = ResidualHead(in_dim, cfg.act_dim).to(device)
    reward_projector = RewardFeatProjector(
        in_dim=reward_model.net[0].out_features,
        out_dim=cfg.reward_feat_dim,
    ).to(device)
    trainable = list(res_model.parameters()) + list(reward_projector.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

    if cfg.watch_model:
        wandb.watch(res_model, log="all", log_freq=cfg.log_every, log_graph=False)

    global_step = 0
    is_reward = (cfg.mode == "reward")

    for epoch in range(1, cfg.epochs + 1):
        res_model.train()
        reward_projector.train()
        epoch_loss, epoch_steps = 0.0, 0

        batches = (train_loader if is_reward
                   else train_loader.iter_batches(batch_size=cfg.batch_size, shuffle=True))

        for batch in batches:
            global_step += 1
            epoch_steps += 1

            # ── unpack batch & compute weights ───────────────────────────────
            if is_reward:
                dp_obs, action_gt, reward = batch
                dp_obs    = {k: v.to(device) if torch.is_tensor(v) else v for k, v in dp_obs.items()}
                action_gt = action_gt.to(device)
                w = make_weights_from_reward(
                    reward.to(device), run_meta["reward_mean"], run_meta["reward_std"], beta=cfg.beta,
                )
            else:
                dp_obs    = batch["dp_obs"]
                action_gt = batch["action"].to(device).float()
                if cfg.mode == "topk":
                    w = torch.ones(action_gt.size(0), device=device)
                else:
                    w_raw = batch["weight"]
                    w = (w_raw.to(device).float().view(-1)
                         if w_raw is not None
                         else torch.ones(action_gt.size(0), device=device))

            # ── forward ──────────────────────────────────────────────────────
            with torch.no_grad():
                z_cond    = dp_model.model.encoder(dp_obs)
                action_dp = dp_sample_with_z(dp_model, z_cond, device)
                _, r_feat = reward_model(z_cond, action_dp, return_feat=True)

            res_in = torch.cat([z_cond, action_dp, reward_projector(r_feat)], dim=-1)

            delta        = clip_by_norm(res_model(res_in), cfg.delta_norm_max)
            target_delta = action_gt - action_dp
            loss_bc      = ((delta - target_delta) ** 2).mean(dim=-1)
            loss         = (w * loss_bc).mean() + cfg.delta_reg * delta.pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

            if global_step % cfg.log_every == 0:
                # diagnostics only — compute lazily to avoid per-step GPU syncs
                with torch.no_grad():
                    action_final  = (action_dp + cfg.lam * delta).clamp(-1.0, 1.0)
                    loss_bc_final = ((action_final - action_gt) ** 2).mean()
                    w_clip_frac   = float((w >= MAX_WEIGHT - 1e-6).float().mean())
                log_dict = {
                    "trainer/step":    global_step,
                    "loss/total":      loss.item(),
                    "loss/bc":         loss_bc.mean().item(),
                    "loss/bc_final":   loss_bc_final.item(),
                    "w/mean":          w.mean().item(),
                    "w/max":           w.max().item(),
                    "w/std":           w.std().item(),
                    "w/clip_frac":     w_clip_frac,
                    "delta/norm_mean": delta.norm(dim=-1).mean().item(),
                }
                if cfg.mode == "advantage":
                    A = batch["advantage"].to(device).float().view(-1)
                    log_dict["adv/mean"] = A.mean().item()
                    log_dict["adv/std"]  = A.std().item()
                wandb.log(log_dict)

        save_checkpoint(out_dir / f"last_epoch{epoch}.pt", res_model,
                        {**run_meta, "epoch": epoch, "global_step": global_step},
                        reward_projector)
        print(f"[epoch {epoch}] loss_avg={epoch_loss / max(1, epoch_steps):.4f}")

    return res_model, reward_projector


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@pyrallis.wrap()
def main(cfg: ResidualPolicyConfig):
    torch.manual_seed(cfg.seed)
    device  = cfg.device
    out_dir = setup_wandb(cfg)
    mode    = cfg.mode

    run_meta: Dict[str, Any] = {
        "time":     time.strftime("%Y%m%d_%H%M%S"),
        "save_dir": str(out_dir),
        "device":   device,
        "run_name": cfg.run_name,
        "mode":     mode,
    }

    dp_model = build_diffusion_model_from_ckpt(cfg.dp_ckpt, ema=True, device=device)

    reward_model = build_reward_model_from_ckpt(
        cfg.reward_model_ckpt, cfg.obs_dim, cfg.act_dim, device,
    )
    print(f"[reward_model] loaded from {cfg.reward_model_ckpt}")

    if cfg.use_memmap:
        dataset = build_dataset_from_memmap(
            memmap_root=Path(cfg.memmap_dir),
            fruit=cfg.fruit,
            reward_alpha=cfg.reward_alpha,
            binary_reward=cfg.binary_reward,
        )
    else:
        dataset = build_arrays_from_roots(
            cfg.dataset_path, cfg.fruit, cfg.reward_alpha,
            binary_reward=cfg.binary_reward,
        )

    # ── reward mode ──────────────────────────────────────────────────────────
    if mode == "reward":
        train_dataset = RewardDataset(dataset)
        run_meta["reward_mean"]  = train_dataset.reward_mean
        run_meta["reward_std"]   = train_dataset.reward_std
        run_meta["reward_range"] = train_dataset.reward_range

        train_loader = DataLoader(
            train_dataset, shuffle=True,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        res_model, reward_projector = train(
            train_loader, dp_model, cfg, run_meta, out_dir, reward_model,
        )
        save_checkpoint(out_dir / "final.pt", res_model, run_meta, reward_projector)

    # ── advantage mode ───────────────────────────────────────────────────────
    elif mode == "advantage":
        req_keys = ["observations_img", "observations_force", "actions", "rewards", "terminals"]
        train_set = {k: np.asarray(dataset[k]) for k in req_keys}
        train_set = _add_advantage_weights(train_set, beta=cfg.beta)
        train_loader = ResidualPolicyDataloader(train_set, device=device)

        res_model, reward_projector = train(
            train_loader, dp_model, cfg, run_meta, out_dir, reward_model,
        )
        save_checkpoint(out_dir / "final.pt", res_model, run_meta, reward_projector)

    # ── topk mode ────────────────────────────────────────────────────────────
    elif mode == "topk":
        rewards_all = np.asarray(dataset["rewards"], dtype=np.float32)
        tau = float(np.quantile(rewards_all, cfg.top_rate))
        keep_mask = rewards_all >= tau
        print(f"[topk] threshold={tau:.4f}, keeping {keep_mask.sum()} / {keep_mask.size} frames")

        req_keys = ["observations_img", "observations_force", "actions", "rewards", "terminals"]
        train_set = {k: np.asarray(dataset[k])[keep_mask] for k in req_keys}
        train_loader = ResidualPolicyDataloader(train_set, device=device)

        res_model, reward_projector = train(
            train_loader, dp_model, cfg, run_meta, out_dir, reward_model,
        )
        save_checkpoint(out_dir / "final.pt", res_model, run_meta, reward_projector)

    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose from 'reward', 'advantage', 'topk'.")

    print(f"[done] checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
