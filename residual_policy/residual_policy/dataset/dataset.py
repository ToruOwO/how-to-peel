import re
import csv
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import json
import random
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import os
import time

from residual_policy.utils.reward_utils import task_cfg_map, RewardConfig


class CorruptPKLError(Exception):
    """Raised when a frame pkl file is corrupted / truncated."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# File utilities
# ──────────────────────────────────────────────────────────────────────────────

def _last_idx(p: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", p.name)
    return int(m.group(1)) if m else -1


def list_sorted(dir_path: Path, pattern: str) -> List[Path]:
    files = list(dir_path.glob(pattern))
    files.sort(key=_last_idx)
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Frame I/O and preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def fuse_rgbd(depth: np.ndarray, rgb_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Fuse depth and RGB into a 2-channel [gray, depth_norm] image with 90% random crop.

    Returns: float32 [2, Hc, Wc]
    """
    mask_f = mask.astype(np.float32) if mask is not None else np.ones(rgb_bgr.shape[:2], dtype=np.float32)

    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray * mask_f, (160, 120), interpolation=cv2.INTER_LINEAR)

    depth_m = depth / 10000.0
    depth_m[depth_m > 0.15] = 0.0
    depth_m = cv2.resize(depth_m * mask_f, (160, 120), interpolation=cv2.INTER_LINEAR)
    depth_n = (np.clip(depth_m, 0.0, 0.15) - 0.05) * 15.0

    image = np.stack([gray, depth_n], axis=0).astype(np.float32)  # [2, H, W]

    H, W = image.shape[1], image.shape[2]
    Hc, Wc = int(H * 0.9), int(W * 0.9)
    cy = random.randint(0, H - Hc - 1) if H > Hc else 0
    cx = random.randint(0, W - Wc - 1) if W > Wc else 0
    return image[:, cy:cy+Hc, cx:cx+Wc]


def read_frame_pkl(pkl_path: Path):
    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise CorruptPKLError(f"Corrupt pickle: {pkl_path}") from e

    full_obs = data["observation"]
    force    = np.asarray(full_obs["force"]).astype(np.float32)
    force_hi = np.array([10., 10., 10., 1., 1., 1.])
    obs = {
        "wrist_mask":  np.asarray(full_obs["wrist_mask"]["binary_mask"]).astype(bool),
        "wrist2_mask": np.asarray(full_obs["wrist2_mask"]["binary_mask"]).astype(bool),
        "force":       normalize(force, -force_hi, force_hi).astype(np.float32),
    }

    act_hi = np.full(6, 0.005)
    action = np.concatenate([
        np.asarray(data["action"]["eef_delta_pos"]).astype(np.float32),
        np.asarray(data["action"]["eef_delta_rot"]).astype(np.float32),
    ])
    action = normalize(action, -act_hi, act_hi).astype(np.float32)
    return obs, action


def read_labels_csv(labels_csv: Path) -> List[int]:
    labels = []
    with labels_csv.open("r") as f:
        for row in csv.DictReader(f):
            if "label" in row:
                labels.append(int(row["label"]))
            elif "final" in row:
                labels.append(int(row["final"]))
    return labels


def normalize(x, lo, hi):
    assert np.all(hi > lo)
    return ((x - lo) / (hi - lo) - 0.5) * 2


def denormalize(x, lo, hi):
    assert np.all(hi > lo)
    return (x + 1) / 2 * (hi - lo) + lo


def smooth_between_segments(rew_arr: np.ndarray, seg_len: int = 15, overlap: int = 3):
    T = len(rew_arr)
    for k in range((T + seg_len - 1) // seg_len - 1):
        end_k      = min(T, (k + 1) * seg_len) - 1
        start_next = (k + 1) * seg_len
        for i in range(overlap):
            a, b = end_k - i, start_next + i
            if b >= T or a < 0:
                continue
            alpha       = (i + 1) / (overlap + 1)
            r_avg       = (1 - alpha) * rew_arr[a] + alpha * rew_arr[b]
            rew_arr[a]  = rew_arr[b] = r_avg


# ──────────────────────────────────────────────────────────────────────────────
# Per-timestamp helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_frames(ts_dir: Path):
    """Load and fuse all frames from one timestamp directory.

    Returns (obs_img [T,4,Hc,Wc], obs_force [T,6], actions [T,6]) or None on error.
    """
    frame_pkls = list_sorted(ts_dir, "data_*.pkl")
    if not frame_pkls:
        return None

    pkl_obs_list, act_list = [], []
    try:
        for pkl_path in frame_pkls:
            obs, act = read_frame_pkl(pkl_path)
            pkl_obs_list.append(obs)
            act_list.append(act)
    except CorruptPKLError as e:
        print(f"[WARN] Skip {ts_dir}: {e}")
        return None

    wrist_rgb    = list_sorted(ts_dir / "wrist_rgb",    "*.png")
    wrist_depth  = list_sorted(ts_dir / "wrist_depth",  "*.npy")
    wrist2_rgb   = list_sorted(ts_dir / "wrist2_rgb",   "*.png")
    wrist2_depth = list_sorted(ts_dir / "wrist2_depth", "*.npy")

    n = min(len(pkl_obs_list), len(wrist_rgb), len(wrist_depth), len(wrist2_rgb), len(wrist2_depth))
    if n == 0 or n != len(pkl_obs_list):
        return None

    imgs, forces = [], []
    for i in range(n):
        pk     = pkl_obs_list[i]
        fused1 = fuse_rgbd(np.load(str(wrist_depth[i]),  mmap_mode="r"), read_image(wrist_rgb[i]),  pk["wrist_mask"])
        fused2 = fuse_rgbd(np.load(str(wrist2_depth[i]), mmap_mode="r"), read_image(wrist2_rgb[i]), pk["wrist2_mask"])
        imgs.append(np.concatenate([fused1, fused2], axis=0))
        forces.append(pk["force"])

    return (
        np.stack(imgs,     axis=0).astype(np.float32),  # [T, 4, Hc, Wc]
        np.stack(forces,   axis=0).astype(np.float32),  # [T, 6]
        np.stack(act_list, axis=0).astype(np.float32),  # [T, 6]
    )


def _process_timestamp(ts_dir: Path, reward_config: RewardConfig):
    """Load frames and compute rewards for one timestamp.

    Returns (obs_img, obs_force, actions, rewards, terminals, next_obs_img, next_obs_force) or None.
    """
    label_csv = ts_dir / "label_1023.csv"
    if not label_csv.exists():
        return None
    seg_labels   = read_labels_csv(label_csv)
    global_label = int(ts_dir.parent.name)

    frames = _load_frames(ts_dir)
    if frames is None:
        return None
    obs_img, obs_force, actions = frames
    T = obs_img.shape[0]

    seg_len = reward_config.get_segment_len()
    overlap  = reward_config.get_overlap()
    rewards  = np.zeros((T,), dtype=np.float32)
    for k in range((T + seg_len - 1) // seg_len):
        label = seg_labels[k] if k < len(seg_labels) else 5
        r = reward_config.get_combined_reward(local_label=label, global_label=global_label, threshold=0.1)
        rewards[k * seg_len : min(T, (k + 1) * seg_len)] += r
    smooth_between_segments(rewards, seg_len=seg_len, overlap=overlap)

    terminals      = np.zeros((T,), dtype=np.float32)
    terminals[-1]  = 1.0
    next_obs_img   = np.concatenate([obs_img[1:],   obs_img[-1:]],   axis=0)
    next_obs_force = np.concatenate([obs_force[1:], obs_force[-1:]], axis=0)

    return obs_img, obs_force, actions, rewards, terminals, next_obs_img, next_obs_force


def _process_timestamp_for_cache(ts_dir: Path, label_csv_name: str = "label_1023.csv"):
    """Like _process_timestamp but skips reward computation; returns raw seg_labels for later caching.

    Returns (obs_img, obs_force, actions, terminals, next_obs_img, next_obs_force, seg_labels, global_label) or None.
    """
    label_csv = ts_dir / label_csv_name
    if not label_csv.exists():
        return None
    seg_labels   = np.array(read_labels_csv(label_csv), dtype=np.int32)
    global_label = int(ts_dir.parent.name)

    frames = _load_frames(ts_dir)
    if frames is None:
        return None
    obs_img, obs_force, actions = frames
    T = obs_img.shape[0]

    terminals      = np.zeros((T,), dtype=np.float32)
    terminals[-1]  = 1.0
    next_obs_img   = np.concatenate([obs_img[1:],   obs_img[-1:]],   axis=0)
    next_obs_force = np.concatenate([obs_force[1:], obs_force[-1:]], axis=0)

    return obs_img, obs_force, actions, terminals, next_obs_img, next_obs_force, seg_labels, global_label


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ──────────────────────────────────────────────────────────────────────────────

def build_arrays_from_roots(roots: List[Path], fruit: str, reward_alpha: float,
                            binary_reward: bool = False) -> Dict[str, np.ndarray]:
    roots = [Path(r) for r in roots]
    all_ts_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"Warning: root {root} not found, skipping.")
            continue
        for score_dir in root.iterdir():
            if not score_dir.is_dir():
                continue
            for ts_dir in score_dir.iterdir():
                if ts_dir.is_dir():
                    all_ts_dirs.append(ts_dir)

    if not all_ts_dirs:
        raise ValueError("No timestamp directories found.")

    reward_cfg = task_cfg_map(fruit.lower(), reward_alpha, binary=binary_reward)
    keys = ["observations_img", "observations_force", "actions", "rewards", "terminals",
            "next_observations_img", "next_observations_force"]
    bufs: Dict[str, list] = {k: [] for k in keys}
    n_loaded = 0

    with ThreadPoolExecutor(max_workers=min(32, len(all_ts_dirs))) as ex:
        futures = {ex.submit(_process_timestamp, ts, reward_cfg): ts for ts in all_ts_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading timestamps"):
            out = fut.result()
            if out is None:
                continue
            for k, arr in zip(keys, out):
                bufs[k].append(arr)
            n_loaded += 1

    if not bufs["observations_img"]:
        raise ValueError("No valid data found in the provided roots.")

    dataset = {k: np.concatenate(v, axis=0) for k, v in bufs.items()}
    print(f"Loaded {dataset['observations_img'].shape[0]} frames from {n_loaded} timestamps.")
    return dataset


def _compute_traj_reward_worker(
    args: Tuple[int, int, int, np.ndarray, int, str, float, bool, int, int]
):
    i, s, e, seg_labels_i, global_label_i, fruit, reward_alpha, binary_reward, seg_len, overlap = args
    T = e - s + 1
    if T <= 0:
        return i, s, e, np.zeros(0, dtype=np.float32)

    reward_cfg = task_cfg_map(fruit.lower(), reward_alpha, binary=binary_reward)
    rewards    = np.zeros((T,), dtype=np.float32)
    for k in range((T + seg_len - 1) // seg_len):
        label = int(seg_labels_i[k]) if k < len(seg_labels_i) else 5
        r     = reward_cfg.get_combined_reward(local_label=label, global_label=global_label_i, threshold=0.1)
        rewards[k * seg_len : min(T, (k + 1) * seg_len)] += r
    smooth_between_segments(rewards, seg_len=seg_len, overlap=overlap)
    return i, s, e, rewards


def _build_rewards_memmap(
    memmap_root: Path,
    rewards_path: Path,
    total_frames: int,
    fruit: str,
    reward_alpha: float,
    binary_reward: bool = False,
):
    traj_slices        = np.load(memmap_root / "traj_slices.npy")
    traj_seg_labels    = np.load(memmap_root / "traj_seg_labels.npy",    allow_pickle=True)
    traj_global_labels = np.load(memmap_root / "traj_global_labels.npy", allow_pickle=True)

    reward_cfg = task_cfg_map(fruit.lower(), reward_alpha, binary=binary_reward)
    seg_len    = reward_cfg.get_segment_len()
    overlap    = reward_cfg.get_overlap()

    rewards_mm = np.memmap(rewards_path, dtype=np.float32, mode="w+", shape=(total_frames,))
    rewards_mm[:] = 0.0

    tasks = [
        (i, int(s), int(e),
         np.asarray(traj_seg_labels[i], dtype=np.int32), int(traj_global_labels[i]),
         fruit, reward_alpha, binary_reward, seg_len, overlap)
        for i, (s, e) in enumerate(traj_slices)
    ]

    with Pool(processes=min(cpu_count(), 8)) as pool:
        for i, s, e, rew_arr in tqdm(
            pool.imap_unordered(_compute_traj_reward_worker, tasks),
            total=len(tasks),
            desc="Building rewards",
        ):
            if rew_arr.size > 0:
                rewards_mm[s:e+1] = rew_arr

    rewards_mm.flush()
    print(f"[memmap] rewards.dat written to {rewards_path}")


def build_dataset_from_memmap(
    memmap_root: str,
    fruit: str,
    reward_alpha: float,
    binary_reward: bool = False,
) -> Dict[str, np.ndarray]:
    memmap_root = Path(memmap_root)
    rank = int(os.environ.get("RANK", "0"))

    with (memmap_root / "meta.json").open("r") as f:
        meta = json.load(f)
    total_frames = int(meta["total_frames"])
    C, H, W = map(int, meta["obs_img_shape"])

    def _mmap_r(name, dtype, shape):
        return np.memmap(memmap_root / name, dtype=dtype, mode="r", shape=shape)

    obs_img_mm   = _mmap_r("observations_img.dat",        np.float16, (total_frames, C, H, W))
    obs_force_mm = _mmap_r("observations_force.dat",      np.float16, (total_frames, 6))
    actions_mm   = _mmap_r("actions.dat",                 np.float16, (total_frames, 6))
    terminals_mm = _mmap_r("terminals.dat",               np.float32, (total_frames,))
    nxt_img_mm   = _mmap_r("next_observations_img.dat",   np.float16, (total_frames, C, H, W))
    nxt_force_mm = _mmap_r("next_observations_force.dat", np.float16, (total_frames, 6))

    # Cache rewards per reward variant so binary/graded (and different alphas)
    # never silently reuse each other's rewards.dat.
    variant = "bin" if binary_reward else "cont"
    alpha_tag = f"{reward_alpha:g}".replace(".", "")
    rewards_name = f"rewards_{variant}_a{alpha_tag}.dat"
    rewards_path = memmap_root / rewards_name
    if not rewards_path.exists():
        if rank == 0:
            print(f"[memmap] building {rewards_name} for {memmap_root} ...")
            _build_rewards_memmap(memmap_root, rewards_path, total_frames,
                                  fruit, reward_alpha, binary_reward)
        else:
            print(f"[rank {rank}] waiting for rank 0 to build {rewards_name} ...")
            while not rewards_path.exists():
                time.sleep(1.0)

    rewards_mm = _mmap_r(rewards_name, np.float32, (total_frames,))

    if rank == 0:
        print(f"[memmap] loaded {total_frames} frames from {memmap_root}")

    return dict(
        observations_img=obs_img_mm,
        observations_force=obs_force_mm,
        actions=actions_mm,
        rewards=rewards_mm,
        terminals=terminals_mm,
        next_observations_img=nxt_img_mm,
        next_observations_force=nxt_force_mm,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Train / val split utilities
# ──────────────────────────────────────────────────────────────────────────────

def split_train_valid_indices(
    dataset: dict,
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by trajectory; return (idx_train, idx_valid) index arrays."""
    terminals = np.asarray(dataset["terminals"]).astype(bool).reshape(-1)
    ends = np.where(terminals)[0]
    if ends.size == 0:
        idx_all = np.arange(dataset["actions"].shape[0], dtype=np.int64)
        return idx_all, np.array([], dtype=np.int64)

    starts     = np.empty_like(ends)
    starts[0]  = 0
    starts[1:] = ends[:-1] + 1
    traj_slices = [np.arange(s, e + 1, dtype=np.int64) for s, e in zip(starts, ends)]

    rng    = np.random.RandomState(seed)
    perm   = rng.permutation(len(traj_slices))
    n_val  = max(1, int(round(valid_ratio * len(traj_slices))))
    val_ids = set(perm[:n_val].tolist())

    train_idx, valid_idx = [], []
    for i, sl in enumerate(traj_slices):
        (valid_idx if i in val_ids else train_idx).append(sl)

    return (
        np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64),
        np.concatenate(valid_idx) if valid_idx else np.array([], dtype=np.int64),
    )


def split_train_valid(
    dataset: dict,
    valid_ratio: float = 0.1,
    seed: int = 0,
    req_keys: Optional[List[str]] = None,
) -> Tuple[dict, dict]:
    """Split dataset by trajectory; return (train_dict, valid_dict)."""
    if req_keys is None:
        req_keys = ["observations_img", "observations_force", "actions", "rewards", "terminals"]
    for k in req_keys:
        if k not in dataset:
            raise KeyError(f"dataset missing key: {k}")

    idx_train, idx_valid = split_train_valid_indices(dataset, valid_ratio, seed)
    return (
        {k: dataset[k][idx_train] for k in req_keys},
        {k: dataset[k][idx_valid] for k in req_keys},
    )


def split_train_valid_steps(
    dataset: dict,
    valid_ratio: float = 0.1,
    seed: int = 0,
    req_keys: Optional[List[str]] = None,
) -> Tuple[dict, dict]:
    """Split by individual steps (no trajectory-boundary awareness)."""
    N    = dataset["actions"].shape[0]
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_val = max(1, min(int(round(valid_ratio * N)), N - 1))

    if req_keys is None:
        req_keys = list(dataset.keys())

    return (
        {k: dataset[k][perm[n_val:]] for k in req_keys},
        {k: dataset[k][perm[:n_val]] for k in req_keys},
    )
