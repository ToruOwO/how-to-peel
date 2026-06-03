from dataclasses import dataclass, field
from typing import Dict

@dataclass
class RewardConfig:
    global_reward: Dict[int, float] = field(default_factory=dict)
    local_reward: Dict[int, float] = field(default_factory=dict)
    alpha: float = 0.8 # weighting between local and global reward
    segment_len: int = 15
    overlap: int = 3 # number of overlapping frames between segments

    def get_local(self, label: int) -> float:
        return self.local_reward.get(label, 0.0)

    def get_global(self, score: int) -> float:
        return self.global_reward.get(score, 0.0)
    
    def get_segment_len(self) -> int:
        return self.segment_len
    
    def get_overlap(self) -> int:
        return self.overlap

    def get_combined_reward(self, local_label: int, global_label: int, threshold: float=0.1) -> float:
        local_r = self.get_local(local_label)
        global_r = self.get_global(global_label)
        if local_r < threshold:
            return local_r
        else:
            return self.alpha * local_r + (1 - self.alpha) * global_r
        
APPLE_BINARY_GLOBAL_REWARD = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}
APPLE_BINARY_LOCAL_REWARD = {
    0: 1.0,   # slightly thin
    1: 1.0,   # perfect
    2: 1.0,   # good
    3: 1.0,   # slightly thick
    4: 0.0,   # too thick
    5: 0.0,   # NA
}

APPLE_GLOBAL_REWARD = {
    0: 0.0,
    1: 0.1,
    2: 0.2,
    3: 0.3,
    4: 0.4,
    5: 0.5,
    6: 0.6,
    7: 0.8,
    8: 0.9,
    9: 1.0,
}
APPLE_LOCAL_REWARD = {
    0: 0.3,   # slightly thin
    1: 1.0,   # perfect
    2: 0.8,   # good
    3: 0.3,   # slightly thick
    4: 0.0,   # too thick
    5: 0.0,   # NA
}

POTATO_BINARY_GLOBAL_REWARD = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}
POTATO_BINARY_LOCAL_REWARD = {
    0: 1.0, # slightly thin
    1: 1.0, # perfect
    2: 1.0, # slightly thick
    3: 0.0, # thick
    4: 0.0, # too thick
    5: 0.0, # NA
}

POTATO_GLOBAL_REWARD = {
    0: 0.0,
    1: 0.1,
    2: 0.2,
    3: 0.3,
    4: 0.4,
    5: 0.5,
    6: 0.6,
    7: 0.8,
    8: 0.9,
    9: 1.0,
}
POTATO_LOCAL_REWARD = {
    0: 0.5, # slightly thin
    1: 1.0, # perfect
    2: 0.5, # slightly thick
    3: 0.1,
    4: 0.0, # too thick
    5: 0.0, # NA
}

# Reward tables per task, indexed by the `binary` flag:
#   (global_reward, local_reward)
_REWARD_TABLES = {
    "apple": {
        False: (APPLE_GLOBAL_REWARD,        APPLE_LOCAL_REWARD),
        True:  (APPLE_BINARY_GLOBAL_REWARD, APPLE_BINARY_LOCAL_REWARD),
    },
    "potato": {
        False: (POTATO_GLOBAL_REWARD,        POTATO_LOCAL_REWARD),
        True:  (POTATO_BINARY_GLOBAL_REWARD, POTATO_BINARY_LOCAL_REWARD),
    },
}


def task_cfg_map(task_name: str, reward_alpha: float, binary: bool = False) -> RewardConfig:
    task_name = task_name.lower()
    if task_name not in _REWARD_TABLES:
        raise ValueError(f"Unknown task name: {task_name!r}. Choose from {list(_REWARD_TABLES)}.")
    global_reward, local_reward = _REWARD_TABLES[task_name][bool(binary)]
    return RewardConfig(
        global_reward=global_reward,
        local_reward=local_reward,
        alpha=reward_alpha,
        segment_len=15,
        overlap=3,
    )
