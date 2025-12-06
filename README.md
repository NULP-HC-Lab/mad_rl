# MAD RL

This repository contains the code for the reinforcement learning pipelines in Isaac Lab.

## Prerequisites

Make sure you have the `mad_assets` repository cloned and available at the same directory
level as this repository.

## Quick start

1. Build the Docker image:
```bash
python docker/container.py start
```

2. Enter the container:
```bash
python docker/container.py enter
```

3. List environments (tasks):
```bash
python scripts/list_envs.py
```

4. Run the training:
```bash
python scripts/rsl_rl/train.py --task <TASK_NAME>
```

5. (Optional) Watch the training curves in real-time inside the container:
```bash
tensorboard --logdir logs
```

## Code formatting

We have a pre-commit to automatically format the code.

To install pre-commit:

```bash
pip install pre-commit
```

Then you can initialize pre-commit with:

```bash
pre-commit install
```
