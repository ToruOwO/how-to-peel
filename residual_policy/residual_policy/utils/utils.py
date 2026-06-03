import os
import numpy as np
import random
import torch


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    

def assert_finite(t, name, step, every=500):
    if step < 1000 or (step % every == 0):
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).float().mean().item()*100
            raise FloatingPointError(f"{name} has non-finite values at step {step}, bad%={bad:.4f}%")
