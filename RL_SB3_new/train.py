import warnings
import os
import subprocess

# First, find all processes that are using the specific port
command = "lsof -i :2000 -t"
processes = subprocess.check_output(command, shell=True).decode().strip().split('\n')

# Now identify which process is not the CARLA server
# For this, you could check the command that started the process, for example:
for pid in processes:
    try:
        # Retrieve the command of the process
        pid_cmd = subprocess.check_output(f"ps -p {pid} -o command=", shell=True).decode().strip()
        # Decide whether to kill based on the command
        if 'CarlaUE4' not in pid_cmd:
            print(f"Killing non-CARLA python process with PID: {pid}")
            subprocess.check_output(f"kill -9 {pid}", shell=True)
    except Exception as e:
        print(f"Failed to check or kill process {pid}: {e}")

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import argparse
import config
import time

parser = argparse.ArgumentParser(description="PPO CARLA agent")
parser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--town", default="Town10HD_Opt", type=str, help="Name of the map in CARLA")
parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total timestep to train for")
parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--num_checkpoints", type=int, default=10, help="Checkpoint frequency")
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")

# Set configuration
args = vars(parser.parse_args())
config.set_config(args["config"])

from stable_baselines3 import PPO, DDPG, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from agent.env import CarlaEnv

#from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

from agent.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

from config import CONFIG
# Create the logging directory
log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)
# Determine the model to load
reload_model = args["reload_model"]
total_timesteps = args["total_timesteps"]

algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
if CONFIG["algorithm"] not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

# Setup environment
env = CarlaEnv(host=args["host"], port=args["port"], town=args["town"],
                fps=args["fps"], obs_sensor_semantic=CONFIG["obs_sensor_semantic"], obs_sensor_rgb=CONFIG["obs_sensor_rgb"], obs_res=CONFIG["obs_res"], 
                    compute_reward=reward_functions[CONFIG["reward_fn"]],
                    view_res=(1200, 600), action_smoothing=CONFIG["action_smoothing"],
                    allow_spectator=True, allow_render=args["no_render"]
                    )

# Load or create the model
if reload_model == "":
    model = AlgorithmRL('MultiInputPolicy', env, verbose=2, tensorboard_log=log_dir, device='cuda',
                        **CONFIG["algorithm_params"])
    model_suffix = f"{int(time.time())}_id{args['config']}"
else:
    model = AlgorithmRL.load(reload_model, env=env, device='cuda', **CONFIG["algorithm_params"])
    model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"

# Setup logging and save config
model_name = f'{model.__class__.__name__}_{model_suffix}'

model_dir = os.path.join(log_dir, model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

# Continue training
model.learn(total_timesteps=total_timesteps,
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                save_freq=total_timesteps // args["num_checkpoints"],
                save_path=model_dir,
                name_prefix="model")], reset_num_timesteps=False)