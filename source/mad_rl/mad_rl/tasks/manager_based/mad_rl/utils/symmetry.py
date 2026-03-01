from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from .constants import (
    HEIGHT_SCAN_NUM_COLS,
    HEIGHT_SCAN_NUM_ROWS,
    LEFT_LEG_JOINT_INDICES,
    OBS_BASE_ANG_VEL_INDICES,
    OBS_BASE_LIN_VEL_INDICES,
    OBS_HEIGHT_SCAN_INDICES,
    OBS_JOINT_POSITIONS_INDICES,
    OBS_JOINT_VELOCITIES_INDICES,
    OBS_PREVIOUS_ACTION_INDICES,
    OBS_PROJECTED_GRAVITY_INDICES,
    OBS_VELOCITY_COMMANDS_INDICES,
    RIGHT_LEG_JOINT_INDICES,
    SYMMETRY_NEGATE_JOINT_INDICES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetry_augmented_data"]


@torch.no_grad()
def compute_symmetry_augmented_data(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment observations and actions by applying left-right symmetry.

    Creates augmented versions of the input data by applying sagittal plane (left-right) symmetry
    transformation. The augmented batch contains both the original and mirrored data, effectively
    doubling the batch size.

    Args:
        env: The environment instance (used for context, not modified).
        obs: Observation tensor dictionary with shape (batch_size, ...).
            If None, observation augmentation is skipped.
        actions: Action tensor with shape (batch_size, num_actions).
            If None, action augmentation is skipped.

    Returns:
        A tuple containing:
            - Augmented observations (2 * batch_size, ...) or None
            - Augmented actions (2 * batch_size, num_actions) or None

        The first half of each output contains original data, the second half contains the mirrored
        version.
    """
    obs_augmented = _augment_observations(obs) if obs is not None else None
    actions_augmented = _augment_actions(actions) if actions is not None else None

    return obs_augmented, actions_augmented


def _augment_observations(obs: TensorDict) -> TensorDict:
    """Create symmetry-augmented observations.

    Args:
        obs: Original observation tensor dictionary.

    Returns:
        Augmented observations with doubled batch size.
    """
    batch_size = obs.batch_size[0]

    # Duplicate batch: [original, mirrored]
    obs_augmented = obs.repeat(2)
    obs_augmented["policy"][:batch_size] = obs["policy"]
    obs_augmented["policy"][batch_size:] = _mirror_policy_observation(obs["policy"])

    return obs_augmented


def _augment_actions(actions: torch.Tensor) -> torch.Tensor:
    """Create symmetry-augmented actions.

    Args:
        actions: Original action tensor.

    Returns:
        Augmented actions with doubled batch size.
    """
    batch_size = actions.shape[0]

    # Duplicate batch: [original, mirrored]
    actions_augmented = actions.repeat(2, 1)
    actions_augmented[batch_size:] = _mirror_actions(actions)

    return actions_augmented


def _mirror_policy_observation(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror transformation to policy observations.

    Transforms observations as if viewed in a mirror along the sagittal plane:
        - Lateral (y) components are negated
        - Roll and yaw rotations are negated
        - Left/right leg joint data is swapped
        - Depth image is flipped horizontally

    Args:
        obs: Policy observation tensor.

    Returns:
        Mirrored observation tensor with the same shape.
    """
    obs = obs.clone()
    device = obs.device

    # Multipliers for sagittal plane reflection
    negate_y = torch.tensor([1.0, -1.0, 1.0], device=device)
    negate_roll_yaw = torch.tensor([-1.0, 1.0, -1.0], device=device)
    negate_lateral_and_yaw = torch.tensor([1.0, -1.0, -1.0], device=device)

    # Base linear velocity: negate lateral (y) component
    obs[:, OBS_BASE_LIN_VEL_INDICES] *= negate_y

    # Base angular velocity: negate roll and yaw, preserve pitch
    obs[:, OBS_BASE_ANG_VEL_INDICES] *= negate_roll_yaw

    # Projected gravity: negate lateral (y) component
    obs[:, OBS_PROJECTED_GRAVITY_INDICES] *= negate_y

    # Velocity commands: negate lateral velocity and yaw rate
    obs[:, OBS_VELOCITY_COMMANDS_INDICES] *= negate_lateral_and_yaw

    # Joint data: swap left/right and negate asymmetric joints
    obs[:, OBS_JOINT_POSITIONS_INDICES] = _mirror_joint_data(obs[:, OBS_JOINT_POSITIONS_INDICES])
    obs[:, OBS_JOINT_VELOCITIES_INDICES] = _mirror_joint_data(obs[:, OBS_JOINT_VELOCITIES_INDICES])
    obs[:, OBS_PREVIOUS_ACTION_INDICES] = _mirror_joint_data(obs[:, OBS_PREVIOUS_ACTION_INDICES])

    if obs.shape[1] > OBS_PREVIOUS_ACTION_INDICES.stop:
        obs[:, OBS_HEIGHT_SCAN_INDICES] = _mirror_height_scan(obs[:, OBS_HEIGHT_SCAN_INDICES])

    return obs


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror transformation to actions.

    Args:
        actions: Action tensor of shape (batch_size, num_actions).

    Returns:
        Mirrored action tensor with the same shape.
    """
    return _mirror_joint_data(actions.clone())


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror transformation to joint data.

    Performs two operations:
        1. Swaps values between left and right leg joints
        2. Negates joints that reverse direction under reflection (hip_yaw, hip_roll, ankle_roll)

    Args:
        joint_data: Joint data tensor of shape (..., 12).

    Returns:
        Mirrored joint data tensor with the same shape.
    """
    mirrored = torch.zeros_like(joint_data)

    # Swap left <-> right leg joints
    mirrored[..., LEFT_LEG_JOINT_INDICES] = joint_data[..., RIGHT_LEG_JOINT_INDICES]
    mirrored[..., RIGHT_LEG_JOINT_INDICES] = joint_data[..., LEFT_LEG_JOINT_INDICES]

    # Negate joints that reverse direction under sagittal reflection
    mirrored[..., SYMMETRY_NEGATE_JOINT_INDICES] *= -1.0

    return mirrored


def _mirror_height_scan(height_scan: torch.Tensor) -> torch.Tensor:
    """Apply left-right mirror transformation to height scan.

    Flips the height scan horizontally to match the sagittal plane symmetry transformation.

    Args:
        height_scan: Height scan tensor.

    Returns:
        Mirrored height scan tensor.
    """
    return (
        height_scan.clone()
        .view(-1, HEIGHT_SCAN_NUM_ROWS, HEIGHT_SCAN_NUM_COLS)
        .flip(dims=[2])
        .view(-1, HEIGHT_SCAN_NUM_ROWS * HEIGHT_SCAN_NUM_COLS)
    )
