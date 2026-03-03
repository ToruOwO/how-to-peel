import json
import pickle
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np

from constants import DATASET_VERSION, LIDAR_ASSET_PATHS, POLICY_CONTROL_FREQ


def write_frames_to_mp4(frames, mp4_path):
    height, width, _ = frames[0].shape
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(str(mp4_path), fourcc, POLICY_CONTROL_FREQ, (width, height))
    for frame in frames:
        # Ensure dtype is acceptable for VideoWriter
        if frame.dtype != np.uint8 and frame.dtype != np.uint16:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()


def read_frames_from_mp4(mp4_path):
    cap = cv.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))
    cap.release()
    return frames


class EpisodeWriter:
    def __init__(self, args, camera_names):
        self.output_dir = Path(args.output_dir)
        self.episode_start_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        self.episode_dir = None

        self.data_config = {k: v for k, v in vars(args).items() if k != "output_dir"}

        # Episode data
        self.timestamps = []
        self.observations = []
        self.actions = []

        self.camera_names = camera_names

        # Write to disk in separate thread to avoid blocking main thread
        self.flush_thread = None

    def step(self, obs, action):
        self.timestamps.append(time.time())
        self.observations.append(obs)
        self.actions.append(action)

    def __len__(self):
        return len(self.observations)

    def _save_lidar_data(self):
        # copy all files from LIDAR_ASSET_PATHS to self.episode_dir
        try:
            for path in LIDAR_ASSET_PATHS:
                shutil.copy(path, self.episode_dir / path.split("/")[-1])
            return True
        except Exception as e:
            print(f"Error saving lidar data: {e}")
            return False

    def _flush(self, score):
        assert len(self) > 0

        self.episode_dir = self.output_dir / f"{score}" / self.episode_start_time
        assert not self.episode_dir.exists()

        # Create episode dir
        self.episode_dir.mkdir(parents=True)

        self._save_lidar_data()

        # Save all image data, rgb in jpeg and depth in npy
        rgbd_keys = []
        for camera_name in self.camera_names:
            rgbd_keys.append(f"{camera_name}_rgb")
            rgbd_keys.append(f"{camera_name}_depth")
        frames_dict = {k: [] for k in rgbd_keys}

        for k in rgbd_keys:
            obs_dir = self.episode_dir / f"{k}"
            obs_dir.mkdir(parents=True)
            for i in range(len(self.observations)):
                if k.endswith("rgb"):
                    cv.imwrite(obs_dir / f"{k}_{i}.png", self.observations[i][k])
                    frames_dict[k].append(self.observations[i][k])
                else:
                    np.save(obs_dir / f"{k}_{i}.npy", self.observations[i][k])
                    depth = cv.applyColorMap(
                        cv.convertScaleAbs(self.observations[i][k], alpha=0.03),
                        cv.COLORMAP_JET,
                    )
                    frames_dict[k].append(depth)

                self.observations[i][k] = None

        # Save other per step data
        for i in range(len(self.observations)):
            with open(self.episode_dir / f"data_{i}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "version": DATASET_VERSION,
                        "timestamp": self.timestamps[i],
                        "observation": self.observations[i],
                        "action": self.actions[i],
                    },
                    f,
                )
        # Save data config as json
        with open(self.episode_dir / "data_config.json", "w") as f:
            json.dump(self.data_config, f)
        num_episodes = len(
            [child for child in self.output_dir.iterdir() if child.is_dir()]
        )
        print(f"Saved episode to {self.episode_dir} ({num_episodes} total)")

        # Write images as MP4 videos
        for k, frames in frames_dict.items():
            mp4_path = self.episode_dir / f"{k}.mp4"
            write_frames_to_mp4(frames, mp4_path)

        # Save segmented masks
        if self.data_config["sam"]:
            for camera_name in self.camera_names:
                mask_frames = []
                for i in range(len(self.observations)):
                    mask = self.observations[i][f"{camera_name}_mask"]["binary_mask"]
                    mask_frame = (
                        (mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                    )
                    if mask_frame.ndim == 2:
                        mask_frame = mask_frame[..., None]
                    mask_frame = mask_frame.repeat(3, axis=-1)
                    mask_frames.append(mask_frame)
                mp4_path = self.episode_dir / f"{camera_name}_mask.mp4"
                write_frames_to_mp4(mask_frames, mp4_path)

        print(f"Wrote all frames to {self.episode_dir}")

    def flush_async(self, score):
        print("Saving successful episode to disk...")
        # Note: Disk writes may cause latency spikes in low-level controllers
        self.flush_thread = threading.Thread(
            target=self._flush, args=(score,), daemon=True
        )
        self.flush_thread.start()

    def wait_for_flush(self):
        if self.flush_thread is not None:
            self.flush_thread.join()
            self.flush_thread = None


class EpisodeReader:
    def __init__(self, episode_dir: str):
        self.episode_dir = Path(episode_dir)

        data_len = len(list(self.episode_dir.glob("data_*.pkl")))

        # Load data
        data = {}
        for i in range(data_len):
            with open(
                self.episode_dir / f"data_{i}.pkl", "rb"
            ) as f:  # Note: Not secure. Only unpickle data you trust.
                data[i] = pickle.load(f)
        self.timestamps = [data[i]["timestamp"] for i in range(data_len)]
        self.observations = [data[i]["observation"] for i in range(data_len)]
        self.actions = [data[i]["action"] for i in range(data_len)]
        assert len(self.timestamps) > 0
        assert len(self.timestamps) == len(self.observations) == len(self.actions)

    def __len__(self):
        return len(self.observations)


if __name__ == "__main__":
    episode_dir = "./data/test_episode"
    reader = EpisodeReader(episode_dir)
    print(len(reader))
