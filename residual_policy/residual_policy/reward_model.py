import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader

from residual_policy.utils.configs import RewardModelConfig
from residual_policy.dataset import (
    build_arrays_from_roots,
    build_dataset_from_memmap,
    split_train_valid_indices,
    RewardDataset,
    collate_fn,
)
from residual_policy.common.models import RewardModel
from diffusion_policy.export import build_diffusion_model_from_ckpt # This API should be by your diffusion policy implementation; adjust the import as needed.


# ---------------- utils ---------------- #
def setup_wandb(cfg: RewardModelConfig) -> Path:
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_name = f"{cfg.run_name}-{timestamp}"
    run = wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        name=str(run_name),
        config=asdict(cfg),
        reinit=True,
        save_code=True,
    )
    wandb.define_metric("trainer/step")
    for k in ["train/*", "val/*", "loss/*"]:
        wandb.define_metric(k, step_metric="trainer/step")

    out_dir = Path(cfg.save_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_checkpoint(path: Path,
                    model: nn.Module,
                    meta: Dict[str, Any]):
    to_save = {
        "reward_model_state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(to_save, path)
    
# ---------------- train & eval ---------------- #

def compute_global_reward_stats(dataset):
    rewards = dataset["rewards"]
    if isinstance(rewards, torch.Tensor):
        r_np = rewards.detach().cpu().numpy().astype("float32")
    else:
        r_np = np.asarray(rewards, dtype=np.float32)

    mean = float(r_np.mean())
    std = float(r_np.std(ddof=0))
    if std < 1e-6:
        std = 1.0
    print(f"[reward stats] mean={mean:.4f}, std={std:.4f}")
    return mean, std

@torch.no_grad()
def compute_pearson(x: Tensor, y: Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    std_x = x.std(unbiased=False)
    std_y = y.std(unbiased=False)
    if std_x < 1e-8 or std_y < 1e-8:
        return float("nan")
    corr = (x * y).mean() / (std_x * std_y + 1e-8)
    return float(corr.item())

@torch.no_grad()
def compute_spearman(x: Tensor, y: Tensor) -> float:
    # Spearman = Pearson(rank(x), rank(y))
    def rank_tensor(t: Tensor) -> Tensor:
        t = t.float()
        sorted_vals, sorted_idx = torch.sort(t)
        ranks = torch.empty_like(sorted_idx, dtype=torch.float32)
        ranks[sorted_idx] = torch.arange(len(t), dtype=torch.float32, device=t.device)
        return ranks

    rx = rank_tensor(x)
    ry = rank_tensor(y)
    return compute_pearson(rx, ry)

@torch.no_grad()
def compute_pairwise_rank_acc(pred: Tensor, target: Tensor, max_pairs: int = 100_000) -> float:
    """
    Approximate pairwise ranking accuracy: sample random pairs (i, j) and
    check whether sign(pred_i - pred_j) agrees with sign(target_i - target_j).
    """
    n = pred.shape[0]
    if n < 2:
        return float("nan")

    device = pred.device
    pred = pred.float()
    target = target.float()

    total_pairs = n * (n - 1) // 2
    num_pairs = min(max_pairs, total_pairs)

    idx_i = torch.randint(0, n, (num_pairs,), device=device)
    idx_j = torch.randint(0, n, (num_pairs,), device=device)

    mask = target[idx_i] != target[idx_j]
    if mask.sum() == 0:
        return float("nan")

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    t_i = target[idx_i]
    t_j = target[idx_j]
    p_i = pred[idx_i]
    p_j = pred[idx_j]

    gt_better = t_i > t_j
    pred_better = p_i > p_j

    acc = (gt_better == pred_better).float().mean()
    return float(acc.item())


@torch.no_grad()
def eval_reward_model(
    model: RewardModel,
    dp_model,
    dataloader,
    device: str,
    mean: float,
    std: float,
    norm_target: bool,
) -> Dict[str, float]:
    model.eval()

    preds = []
    trues = []

    for batch in dataloader:
        dp_obs, action, reward = batch
        reward = reward.to(device)
        for k, v in dp_obs.items():
            if torch.is_tensor(v):
                dp_obs[k] = v.to(device)
        action = action.to(device)

        if norm_target:
            reward = (reward - mean) / std
        else:
            reward = reward

        z_cond = dp_model.model.encoder(dp_obs)   # [B, obs_dim]
        r_pred = model(z_cond, action)              # [B]
        r_pred = r_pred.view(-1)
        
        preds.append(r_pred.detach().cpu())
        trues.append(reward.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    
    val_loss = F.mse_loss(preds, trues).item()
    pearson  = compute_pearson(preds, trues)
    spearman = compute_spearman(preds, trues)
    rank_acc = compute_pairwise_rank_acc(preds, trues, max_pairs=100_000)

    return {
        "val/loss": val_loss,
        "val/pearson": pearson,
        "val/spearman": spearman,
        "val/rank_acc": rank_acc,
    }


def train_reward_model(
    cfg: RewardModelConfig,
    dp_model,
    train_loader,
    valid_loader,
    out_dir: Path,
    r_mean: float,
    r_std: float,
) -> None:
    device = cfg.device
    model = RewardModel(cfg.obs_dim, cfg.act_dim,
                        hidden_dim=cfg.hidden_dim,
                        num_layers=cfg.num_layers).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    if cfg.watch_model:
        wandb.watch(model, log="all", log_freq=cfg.log_every, log_graph=False)

    global_step = 0
    best_val_loss = None
    best_ckpt_path = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            global_step += 1
            epoch_steps += 1

            dp_obs, action, reward = batch
            reward = reward.to(device)
            for k, v in dp_obs.items():
                if torch.is_tensor(v):
                    dp_obs[k] = v.to(device)
            action = action.to(device)

            if cfg.norm_target:
                target = (reward - r_mean) / r_std
            else:
                target = reward

            with torch.no_grad():
                z_cond = dp_model.model.encoder(dp_obs)  # [B, obs_dim]

            pred = model(z_cond, action)                 # [B]
            loss = F.mse_loss(pred, target, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            epoch_loss += loss.item()

            # logging
            if global_step % cfg.log_every == 0:
                with torch.no_grad():
                    mae = torch.mean(torch.abs(pred - target))
                log_dict = {
                    "trainer/step": global_step,
                    "train/loss": loss.item(),
                    "train/mae": mae.item(),
                    "train/reward_mean_batch": reward.mean().item(),
                    "train/reward_std_batch": reward.std(unbiased=False).item(),
                }
                wandb.log(log_dict)

            # eval
            if (global_step % cfg.eval_every == 0) and (valid_loader is not None):
                val_metrics = eval_reward_model(
                    model, dp_model, valid_loader,
                    device=device,
                    mean=r_mean,
                    std=r_std,
                    norm_target=cfg.norm_target,
                )
                val_loss = val_metrics["val/loss"]
                print(
                    f"[epoch {epoch}] train_loss={epoch_loss / max(1, epoch_steps):.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"pearson={val_metrics['val/pearson']:.4f} "
                    f"spearman={val_metrics['val/spearman']:.4f} "
                    f"rank_acc={val_metrics['val/rank_acc']:.4f}"
                )
                val_metrics["trainer/step"] = global_step
                wandb.log(val_metrics)

                current = val_metrics["val/loss"]
                if (best_val_loss is None) or (current < best_val_loss):
                    best_val_loss = current
                    best_ckpt_path = out_dir / f"best_epoch{epoch}_step{global_step}_loss{best_val_loss:.6f}.pt"
                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        meta={
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_metrics": val_metrics,
                            "reward_mean": r_mean,
                            "reward_std": r_std,
                        },
                    )
                    wandb.log({"model/best_improved": 1, "trainer/step": global_step})
                else:
                    wandb.log({"model/best_improved": 0, "trainer/step": global_step})

        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}")

        last_path = out_dir / f"last_epoch{epoch}.pt"
        save_checkpoint(
            last_path,
            model,
            meta={
                "epoch": epoch,
                "global_step": global_step,
                "reward_mean": r_mean,
                "reward_std": r_std,
                "last_train_loss": avg_loss,
            },
        )

    final_val = {}
    if valid_loader is not None:
        final_val = eval_reward_model(
            model, dp_model, valid_loader,
            device=device,
            mean=r_mean,
            std=r_std,
            norm_target=cfg.norm_target,
        )
        final_val["trainer/step"] = global_step
        wandb.log(final_val)

    final_path = out_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        meta={
            "epoch": cfg.epochs,
            "global_step": global_step,
            "reward_mean": r_mean,
            "reward_std": r_std,
            "final_val": final_val,
        },
    )
    print(f"[final] saved to {final_path}")
    if best_ckpt_path is not None:
        print(f"[best ] val/loss={best_val_loss:.6f} at {best_ckpt_path}")


# ---------------- main ---------------- #
@pyrallis.wrap()
def main(cfg: RewardModelConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    out_dir = setup_wandb(cfg)

    dp_model = build_diffusion_model_from_ckpt(
        cfg.dp_ckpt,
        ema=True,
        device=device,
    )

    if cfg.use_memmap:
        dataset = build_dataset_from_memmap(
            memmap_root=Path(cfg.memmap_dir),
            fruit=cfg.fruit,
            reward_alpha=cfg.reward_alpha,
            binary_reward=cfg.binary_reward,
        )
    else:
        dataset = build_arrays_from_roots(
            cfg.dataset_path,
            cfg.fruit,
            cfg.reward_alpha,
            binary_reward=cfg.binary_reward,
        )

    r_mean, r_std = compute_global_reward_stats(dataset)
    r_mean_t = torch.tensor(r_mean, device=device, dtype=torch.float32)
    r_std_t = torch.tensor(r_std, device=device, dtype=torch.float32)

    idx_train, idx_valid = split_train_valid_indices(
        dataset,
        valid_ratio=cfg.valid_ratio,
        seed=cfg.seed,
    )
    train_dataset = RewardDataset(
        dataset=dataset,
        indices=idx_train,
    )
    full_dataset = RewardDataset(
        dataset=dataset
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        full_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    train_reward_model(
        cfg=cfg,
        dp_model=dp_model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        out_dir=out_dir,
        r_mean=r_mean_t,
        r_std=r_std_t,
    )


if __name__ == "__main__":
    main()