import warnings
import os
import subprocess
import time
# First, find all processes that are using the specific port
command = "lsof -i :2000 -t"
try:
    processes = subprocess.check_output(command, shell=True).decode().strip().split('\n')
except subprocess.CalledProcessError as e:
    # Handle the case where no processes are found or another error occurs
    print("No processes found using port 2000, or an error occurred.")
    processes = []
print(processes)

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
# Start CARLA Server

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import config
import time

parser = argparse.ArgumentParser(description="Evaluate PPO CARLA agent")
parser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--town", default="Town10HD_Opt", type=str, help="Name of the map in CARLA")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
parser.add_argument("--reload_model", type=str, required=True, help="Path to a model to load for evaluation")
parser.add_argument("--no_render", action="store_true", help="If set, do not render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")

args = vars(parser.parse_args())
config.set_config(args["config"])

from stable_baselines3 import PPO, SAC
from agent.env import CarlaEnv
from agent.rewards import reward_functions
from config import CONFIG

# Setup environment
env = CarlaEnv(host=args["host"], port=args["port"], town=args["town"],
                fps=args["fps"], obs_sensor_semantic=CONFIG["obs_sensor_semantic"], obs_sensor_rgb=CONFIG["obs_sensor_rgb"], obs_res=CONFIG["obs_res"], 
                reward_fn=reward_functions[CONFIG["reward_fn"]],
                view_res=(1200, 600), action_smoothing=CONFIG["action_smoothing"],
                allow_spectator=True, allow_render=not args["no_render"])

# Load the model
model = PPO.load(args["reload_model"], env=env)

# Evaluate the model
episode_rewards = []
for episode in range(args["num_episodes"]):
    obs = env.reset()
    done = False
    total_rewards = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
        if args["no_render"]:
            env.render()
    episode_rewards.append(total_rewards)
    print(f"Episode {episode + 1}: Total Reward: {total_rewards}")

print("Average reward:", sum(episode_rewards) / len(episode_rewards))
