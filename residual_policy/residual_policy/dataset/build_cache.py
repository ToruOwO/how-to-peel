import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from residual_policy.dataset.dataset import _process_timestamp_for_cache


def collect_all_ts_dirs(roots: List[Path]) -> List[Path]:
    roots = [Path(r) for r in roots]
    all_ts = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] root {root} does not exist, skip.")
            continue
        for score_dir in root.iterdir():
            if not score_dir.is_dir():
                continue
            for ts_dir in score_dir.iterdir():
                if ts_dir.is_dir():
                    all_ts.append(ts_dir)
    return all_ts


def first_pass_get_shape_and_lengths(
    ts_dirs: List[Path],
    label_csv: str,
) -> Tuple[List[Tuple[Path, int]], int, Tuple[int, int, int]]:
    ts_infos = []
    total_frames = 0
    obs_img_shape = None  # (C, H, W)

    for ts_dir in tqdm(ts_dirs, desc="First pass (collect lengths)"):
        out = _process_timestamp_for_cache(ts_dir, label_csv)
        if out is None:
            continue
        oi, of, ac, tm, ni, nf, seg_labels, global_label = out
        T = oi.shape[0]
        if T == 0:
            continue

        if obs_img_shape is None:
            # oi: [T, C, H, W]
            obs_img_shape = oi.shape[1:]  # (C, H, W)

        ts_infos.append((ts_dir, T))
        total_frames += T

    if not ts_infos:
        raise ValueError("No valid timestamps after first pass.")

    return ts_infos, total_frames, obs_img_shape


def build_memmap_dataset(
    roots: List[str],
    label_csv: str,
    out_dir: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(r) for r in roots]
    all_ts_dirs = collect_all_ts_dirs(roots)

    ts_infos, total_frames, obs_img_shape = first_pass_get_shape_and_lengths(all_ts_dirs, label_csv)
    C, H, W = obs_img_shape
    print(f"[memmap] total_frames={total_frames}, obs_img_shape={obs_img_shape}, num_ts={len(ts_infos)}")

    obs_img_mm = np.memmap(out_dir / "observations_img.dat",
                           dtype=np.float16,
                           mode="w+",
                           shape=(total_frames, C, H, W))
    obs_force_mm = np.memmap(out_dir / "observations_force.dat",
                             dtype=np.float16,
                             mode="w+",
                             shape=(total_frames, 6))
    actions_mm = np.memmap(out_dir / "actions.dat",
                           dtype=np.float16,
                           mode="w+",
                           shape=(total_frames, 6))
    terminals_mm = np.memmap(out_dir / "terminals.dat",
                             dtype=np.float32,
                             mode="w+",
                             shape=(total_frames,))
    next_obs_img_mm = np.memmap(out_dir / "next_observations_img.dat",
                                dtype=np.float16,
                                mode="w+",
                                shape=(total_frames, C, H, W))
    next_obs_force_mm = np.memmap(out_dir / "next_observations_force.dat",
                                  dtype=np.float16,
                                  mode="w+",
                                  shape=(total_frames, 6))

    traj_slices = []  # list of (start_idx, end_idx)
    traj_ts_dirs = []
    traj_seg_labels = []      # list of np.ndarray
    traj_global_labels = []   # list of int

    cur = 0
    for ts_dir, T in tqdm(ts_infos, desc="Second pass (write memmap)"):
        out = _process_timestamp_for_cache(ts_dir, label_csv)
        if out is None:
            continue
        oi, of, ac, tm, ni, nf, seg_labels, global_label = out
        T_real = oi.shape[0]
        if T_real == 0:
            continue

        s, e = cur, cur + T_real - 1
        traj_slices.append((s, e))
        traj_ts_dirs.append(str(ts_dir))

        obs_img_mm[s:e+1] = oi.astype(np.float16)
        obs_force_mm[s:e+1] = of.astype(np.float16)
        actions_mm[s:e+1] = ac.astype(np.float16)
        terminals_mm[s:e+1] = tm.astype(np.float32)
        next_obs_img_mm[s:e+1] = ni.astype(np.float16)
        next_obs_force_mm[s:e+1] = nf.astype(np.float16)
        
        traj_seg_labels.append(seg_labels.astype(np.int16))
        traj_global_labels.append(int(global_label))

        cur = e + 1

    for mm in (obs_img_mm, obs_force_mm, actions_mm, terminals_mm,
               next_obs_img_mm, next_obs_force_mm):
        mm.flush()

    used_frames = cur
    print(f"[memmap] used_frames={used_frames}")

    meta = {
        "total_frames": int(used_frames),
        "obs_img_shape": [int(C), int(H), int(W)],
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    np.save(out_dir / "traj_slices.npy", np.asarray(traj_slices, dtype=np.int64))
    np.save(out_dir / "traj_ts_dirs.npy", np.asarray(traj_ts_dirs, dtype=object))
    np.save(out_dir / "traj_seg_labels.npy", np.array(traj_seg_labels, dtype=object))
    np.save(out_dir / "traj_global_labels.npy", np.array(traj_global_labels, dtype=np.int32))

    print(f"[memmap] done, saved in {out_dir}")


if __name__ == "__main__":
    build_memmap_dataset(
        roots=[],
        label_csv="label.csv",
        out_dir="./memmap_out",
    )
