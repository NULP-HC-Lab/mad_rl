import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one
    foot at a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is
    zero.

    Note: This function is a modified version of the original function from the Isaac Lab Velocity
    MDP. It takes separate contact sensors for left and right foot and merges the data.
    """
    # get sensors for left and right foot
    left_sensor: ContactSensor = env.scene.sensors[left_foot_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_foot_sensor_cfg.name]
    # merge air time and contact time from both feet
    air_time = torch.cat(
        [
            left_sensor.data.current_air_time[:, left_foot_sensor_cfg.body_ids],
            right_sensor.data.current_air_time[:, right_foot_sensor_cfg.body_ids],
        ],
        dim=1,
    )
    contact_time = torch.cat(
        [
            left_sensor.data.current_contact_time[:, left_foot_sensor_cfg.body_ids],
            right_sensor.data.current_contact_time[:, right_foot_sensor_cfg.body_ids],
        ],
        dim=1,
    )
    # compute the reward
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env,
    left_foot_sensor_cfg: SceneEntityCfg,
    right_foot_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as
    the norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures
    that the agent is penalized only when the feet are in contact with the ground.

    Note: This function is a modified version of the original function from the Isaac Lab Velocity
    MDP. It takes separate contact sensors for left and right foot and merges the data.
    """
    # get sensors for left and right foot
    left_sensor: ContactSensor = env.scene.sensors[left_foot_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_foot_sensor_cfg.name]
    # merge net forces from both feet
    net_forces = torch.cat(
        [
            left_sensor.data.net_forces_w_history[:, :, left_foot_sensor_cfg.body_ids, :],
            right_sensor.data.net_forces_w_history[:, :, right_foot_sensor_cfg.body_ids, :],
        ],
        dim=2,
    )
    contacts = net_forces.norm(dim=-1).max(dim=1)[0] > 1.0
    # get body velocities for both feet
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def jump(
    env: ManagerBasedRLEnv,
    left_foot_contact_sensor_cfg: SceneEntityCfg,
    right_foot_contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize jumping."""
    left_sensor: ContactSensor = env.scene.sensors[left_foot_contact_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_foot_contact_sensor_cfg.name]
    air_time = torch.cat(
        [
            left_sensor.data.current_air_time,
            right_sensor.data.current_air_time,
        ],
        dim=1,
    )
    return air_time.min(dim=1)[0]
