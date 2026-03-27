# MAD RL

This repository contains the code for the reinforcement learning pipelines in Isaac Lab.

## Prerequisites

- Make sure you have the `mad_assets` repository cloned and available at the same
directory level as this repository.

- Install Pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

## Quick start

1. Build the Docker image:
```bash
docker compose -p mad build
```

2. Start the container:
```bash
docker compose -p mad up --watch
```

3. Enter the container:
```bash
docker compose -p mad exec rl bash
```

3. List environments (tasks):
```bash
pixi r list_envs
```

4. Run the training:
```bash
pixi r train --task <TASK_NAME>
```

5. (Optional) Watch the training curves in real-time inside the container:
```bash
pixi r tensorboard
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
