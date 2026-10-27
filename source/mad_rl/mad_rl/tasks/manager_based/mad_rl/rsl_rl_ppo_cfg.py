from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from .utils.symmetry import compute_symmetry_augmented_data


@configclass
class RslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 40_000
    save_interval = 500
    experiment_name = "mad_rl"
    empirical_normalization = False
    # "rsl-rl>=4" renames "policy"/"critic" to "actor"/"critic". 
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    # "rsl-rl>=4" splits the old ActorCritic config with general model config. 
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_symmetry_augmented_data,
        ),
    )
