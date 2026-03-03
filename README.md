# 🔪 How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference

[[Project](https://toruowo.github.io/peel/)]
[[arXiv](https://arxiv.org/abs/2404.16823)]
[[Video](https://arxiv.org/abs/2404.16823)]

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
- learning a reward model from human preference (coming soon)
- finetuning Diffusion Policies with a learned reward model (coming soon)
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

## Datasets [COMING SOON]

[[Data Files](https://www.dropbox.com/scl/fo/3kcq7kreifmpgbdsjma1s/APeYN8EOhRYcSvAO0w0ajl4?rlkey=pxgmd6ur81v6wj2urdod3ld82&dl=0)]

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

## Finetuning [COMING SOON]

### learning a reward model from human preference

### finetuning base policy with a reward model

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
    journal={arXiv:xxxx.xxxxx},
    year={2026}
}
```