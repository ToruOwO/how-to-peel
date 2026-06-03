import torch
from torch.utils.data import Dataset
import numpy as np


class ResidualPolicyDataloader:
    def __init__(self, dataset: dict, device: str):
        self.device = device

        self.image1     = torch.from_numpy(dataset["observations_img"][:, :2]).float()
        self.image2     = torch.from_numpy(dataset["observations_img"][:, 2:]).float()
        self.states     = torch.from_numpy(dataset["observations_force"]).float()
        self.actions    = torch.from_numpy(dataset["actions"]).float()
        self.advantages = torch.from_numpy(dataset["advantages"]).float().view(-1, 1) if "advantages" in dataset else None
        self.weights    = torch.from_numpy(dataset["weights"]).float().view(-1, 1) if "weights" in dataset else None

        if device.startswith("cuda"):
            for attr, val in vars(self).items():
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.pin_memory())

        self.N = self.actions.size(0)

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        order = torch.randperm(self.N) if shuffle else torch.arange(self.N)
        for s in range(0, self.N, batch_size):
            idx = order[s: s + batch_size]
            sel = lambda t: t.index_select(0, idx).to(self.device, non_blocking=True)

            yield {
                "dp_obs":    {"image1": sel(self.image1), "image2": sel(self.image2), "state": sel(self.states)},
                "action":    sel(self.actions),
                "advantage": sel(self.advantages) if self.advantages is not None else None,
                "weight":    sel(self.weights) if self.weights is not None else None,
            }


class RewardDataset(Dataset):
    def __init__(self, 
        dataset: dict,
        indices: np.ndarray = None,
    ):
        super().__init__()
        
        self.img    = dataset["observations_img"]      # [N, 4, H, W]
        self.force  = dataset["observations_force"]    # [N, 6]
        self.action = dataset["actions"]               # [N, 6]
        self.reward = dataset["rewards"]               # [N,]

        if indices is None:
            self.indices = np.arange(self.action.shape[0], dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        reward_np = self.reward[self.indices].astype("float32")  # [M]
        self.reward_mean = float(np.nanmean(reward_np))
        self.reward_std  = float(np.nanstd(reward_np) + 1e-8)
        reward_max = float(np.nanmax(reward_np))
        reward_min = float(np.nanmin(reward_np))
        self.reward_range = (reward_min, reward_max)
        print(f"[Dataset] reward mean={self.reward_mean:.4f}, std={self.reward_std:.4f}, "
              f"reward max={np.nanmax(reward_np):.4f}, reward min={np.nanmin(reward_np):.4f}")
      
        self.N = self.indices.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        force = self.force[real_idx]    # [6]
        act = self.action[real_idx]     # [6]
        img = self.img[real_idx]        # [4, H, W]
        img1 = img[:2]   # [2, H, W]
        img2 = img[2:]   # [2, H, W]
        
        rew = torch.tensor(self.reward[real_idx], dtype=torch.float32)

        cond = {
            "image1": torch.tensor(img1, dtype=torch.float32),
            "image2": torch.tensor(img2, dtype=torch.float32),
            "state":  torch.tensor(force, dtype=torch.float32),
        }

        y = torch.tensor(act, dtype=torch.float32)

        return cond, y, rew
    
    
def collate_fn(batch):
    cond_list = [b[0] for b in batch]
    y_list = [b[1] for b in batch]

    cond_batched = {}
    for k in cond_list[0].keys():
        cond_batched[k] = torch.stack([c[k] for c in cond_list], dim=0)

    y_batched = torch.stack(y_list, dim=0)

    reward = torch.stack([b[2] for b in batch], dim=0).float()
    return cond_batched, y_batched, reward
