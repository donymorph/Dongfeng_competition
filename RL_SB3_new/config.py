import torch
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

algorithm_params = {
    "PPO": dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 1e-2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                           net_arch=dict(pi=[500, 300], vf=[500, 300]))
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_BEST": dict(
        learning_rate=lr_schedule(5e-7, 3e-4, 1e-3),
        buffer_size=600000,
        batch_size=1024,
        ent_coef='auto',
        gamma=0.99,
        tau=0.01,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[512, 512, 512], use_sde =True),
    ),
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
     "reward_fn_5_no_early_stop": dict(
         early_stop=False,
         min_speed=20.0,  # km/h
         max_speed=35.0,  # km/h
         target_speed=25.0,  # kmh
         max_distance=3.0,  # Max distance from center before terminating
         max_std_center_lane=0.4,
         max_angle_center_lane=90,
         penalty_reward=-10,
     ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=10.0,  # km/h
        max_speed=20.0,  # km/h 
        target_speed=15.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "combined_reward": dict(
        early_stop=True,
        min_speed=5.0,  # km/h
        max_speed=10.0,  # km/h 
        target_speed=7.0,  # kmh
        max_distance=5.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-100,
    ),
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "action_smoothing": 0.00,
    "reward_fn": "combined_reward",
    "reward_params": reward_params["combined_reward"],
    "obs_sensor_rgb": "rgb",
    "obs_sensor_semantic": "semantic",
    "obs_res": (196, 100),
}
CONFIGS = {
    "1": _CONFIG_1
}
CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = CONFIGS[config_name]