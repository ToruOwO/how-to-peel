# 🔪 How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference

[[Project](https://toruowo.github.io/peel/)]
[[arXiv](https://arxiv.org/abs/2603.03280)]
[[Video](https://youtu.be/TBJr34EYYGA?si=DiyBqc2j0L-tBQeG)]

[Toru Lin*](https://toruowo.github.io/),
[Shuying Deng*](https://shuyingdeng.github.io/),
[Zhao-Heng Yin](http://zhaohengyin.github.io/),
[Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)
<br>

## Overview

This repo contains code, datasets, and instructions to support the following use cases:
- teleoperating a Kinova Gen3 arm with impedance control with SpaceMouse, keyboard, or heuristic-based planner
- collecting multimodal demonstration data
- training peeling policies
- learning a reward model from human preference
- finetuning Diffusion Policies with a learned reward model
- deploying policies to real robot hardware

## Dependencies

### hardware

This code base is tailored to a hardware setup with:
- a 7-DoF Kinova Gen3 arm
- an ATI mini45 force-torque sensor
- two RealSense D405 wrist cameras
- a 3Dconnexion SpaceMouse (for teleoperation)

### software

- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time) (we recommend this library over the original SAM2 to support real-time segmentation of video stream)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/#paper) (for training the base policy)
- PyTorch
- Python libraries for Kinova Gen3 arm, RealSense cameras, and SpaceMouse

## Datasets

[[Data Files](https://www.dropbox.com/scl/fo/84cdwlodyfrdqh8rz5o19/ADf2olooDpjAYTBjsYGNfZk?rlkey=0vhe0c0glws3nwnzkdroazjj8&st=w3usc8l2&dl=0)]

The linked data folder contains datasets for the three produce types featured in our paper: `apple`, `potato`, and `cucumber`.

Note that the datasets are split into part files, each with a maximal size of 20G. The part files should be first concatenated back into single files, then extracted using the following commands:

```bash
mkdir apple
cat apple.tar.part-* | tar -xf - -C apple

mkdir potato
cat potato.tar.part-* | tar -xf - -C potato

mkdir cucumber
cat cucumber.tar.part-* | tar -xf - -C cucumber
```

Scripts to download and concatenate the datasets can be found in `download_dataset.sh`.

## Collecting data

```bash
# teleoperate end-effector with SpaceMouse and save collected data in ./demos
python main.py --save --output-dir ./demos --eef

# teleoperate end-effector with SpaceMouse without saving and with real-time segmentation
python main.py --eef --sam
```

Arguments for tests:
- `--test`: run in a mock environment without needing to activate robot or camera
- `--test-camera`: run in a mock environment without needing to activate camera

Arguments for controller:
- `--pos`: use position control for the arm (default is impedance control)

Arguments for teleoperation:
- `--planner`: use heuristic-based planner for teleoperation; this requires an additional third-view L515 camera (default is SpaseMouse teleoperation)
- `--keyboard`: use keyboard for teleoperation (default is SpaseMouse teleoperation)
- `--eef`: use end-effector-frame when calculating delta control command for teleoperation (default is Cartesian-frame, same as arm base)
- `--object-type`: this changes the initial end-effector position for objects with different lengths (can be "long" or "short", default is "long")

Arguments for real-time segmentation:
- `--sam`: turn on real-time segmentation during teleoperation
- `--sam-model-type`: choose model type for SAM2 (default is "b+")
- `--sam-fps`: optionally run SAM2 at a lower frequency than data collection to improve latency

Arguments for data collection:
- `--save`: save data during teleoperation
- `--output-dir`: output directory path to save data

## Training

Please follow the implementation details in paper to modify Diffusion Policy for base policy training.

## Deploying

Arguments for policy deployment:
- `--scripted`: run a scripted policy to test arm movement
- `--replay`: replay trajectory from a specified folder
- `--ckpt-dir`: checkpoint path for policy to be deployed
- `--ema`: deploy policy trained with EMA

## Finetuning

Finetune a pretrained diffusion policy with a lightweight **residual head** that
nudges the base action toward higher-reward behavior, where the reward is learned
from human annotations. The pipeline has two stages:

1. **Learn a reward model** that scores `(observation, action)` pairs, supervised
   by human preference labels.
2. **Finetune the base policy** by training a residual head against that reward
   model, keeping the base diffusion policy frozen.

The final action is `a = a_base + lam * delta`, where `a_base` is sampled from the
frozen base policy and `delta` is produced by the residual head.

### Setup

```bash
cd residual_policy
pip install -r requirements.txt
```

> `build_diffusion_model_from_ckpt` is imported from `diffusion_policy.export`,
> which is your own base diffusion policy implementation. Install it separately
> and adjust the import in `reward_model.py` / `res_policy.py` if your API differs.

Demonstrations and their human labels are first cached into a memmap dataset for
fast loading:

```bash
python -m residual_policy.dataset.build_cache  # edit roots / label_csv / out_dir first
```

### learning a reward model from human preference

Each demonstration carries human annotations — a per-segment local score
and a global score for the whole episode. These are mapped to a scalar reward by
the reward tables in `residual_policy/utils/reward_utils.py` (graded by default,
or binary via `binary_reward: true`). The reward model is an MLP that regresses
this reward from the frozen DP encoder embedding and the action.

```bash
python -m residual_policy.reward_model --config_path configs/reward/apple.yaml
```

Key fields in `configs/reward/*.yaml`:

| field | meaning |
|-------|---------|
| `dp_ckpt` | base diffusion policy checkpoint (its encoder is used, frozen) |
| `memmap_dir` | pre-built memmap dataset directory |
| `fruit` | task name (`apple` / `potato`) selecting the reward table |
| `binary_reward` | use binary (0/1) reward instead of graded |
| `norm_target` | z-score normalize the regression target |

Checkpoints are written to `save_dir`; use the best one as `reward_model_ckpt`
in the next stage. Validation logs Pearson / Spearman correlation and pairwise
ranking accuracy against the human reward.

### finetuning base policy with a reward model

The residual head is trained with behavior cloning weighted by the learned
reward. Three modes are available via `mode` in the config:

- **`reward`** — reward-conditioned residual; the reward model's features are
  projected and concatenated to the input of the residual head.
- **`advantage`** — advantage-weighted BC, weighting each frame by its
  per-trajectory advantage `r_t - mean(r_τ)`.
- **`topk`** — top-K filtered BC, keeping only frames whose reward exceeds the
  `top_rate` quantile.

```bash
python -m residual_policy.res_policy --config_path configs/res/apple_reward.yaml
```

Key fields in `configs/res/*.yaml`:

| field | meaning |
|-------|---------|
| `mode` | `reward` / `advantage` / `topk` |
| `dp_ckpt` | base diffusion policy checkpoint to finetune |
| `reward_model_ckpt` | reward model trained in the previous stage |
| `memmap_dir` | pre-built memmap dataset directory |
| `lam` | residual blend: `a = a_base + lam * delta` |
| `beta` | temperature for exponential reward/advantage weighting |
| `delta_norm_max` | max L2 norm of the residual action |

Checkpoints (residual head + reward feature projector) are saved per epoch under
`save_dir`, with a `final.pt` at the end.


## Acknowledgement

This project was developed with help from the following codebases.

- [tidybot2](https://github.com/jimmyyhwu/tidybot2)
- [compliant_controllers](https://github.com/applied-ai-lab/compliant_controllers)
- [gen3_compliant_controllers](https://github.com/empriselab/gen3_compliant_controllers)

## Reference

If you find our paper or this codebase helpful in your research, please consider citing:

```
@article{lin2026how,
    author={Lin, Toru and Deng, Shuying and Yin, Zhao-Heng and Abbeel, Pieter and Malik, Jitendra},
    title={How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference},
    journal={arXiv:2603.03280},
    year={2026}
}
```
