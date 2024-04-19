import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if speed is less than 1.0 km/h after the first 5s of an episode
            global low_speed_timer
            low_speed_timer += 1.0 / env.fps
            speed = env.get_vehicle_lon_speed()
            if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 0:
                env.terminate = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if env.distance_from_center > max_distance:
                env.terminate = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if max_speed > 0 and speed > max_speed:
                env.terminate = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminate:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func

def calculate_speed_reward(speed_kmh):
    """Calculate a non-linear speed reward."""
    if speed_kmh < min_speed:
        return np.exp(-(min_speed - speed_kmh) / min_speed)
    elif speed_kmh > target_speed:
        if speed_kmh < max_speed:
            return 1.0 - np.exp((target_speed - speed_kmh) / (max_speed - target_speed))
        else:
            return -np.exp((speed_kmh - max_speed) / max_speed)
    return 1.0
    
def reward_fn5(env):
    """Complex reward function considering multiple aspects of driving."""
    speed_kmh = env.get_vehicle_lon_speed()
    speed_reward = calculate_speed_reward(speed_kmh)

    veh_angle = env.vehicle.get_transform().rotation.yaw
    wayp_angle = env.current_waypoint.transform.rotation.yaw
    angle = abs(wayp_angle - veh_angle)

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    reward = speed_reward * centering_factor * angle_factor * distance_std_factor
    return reward

reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)

def reward_fn_waypoints(env):
    """Reward based on waypoint traversal and speed conditions."""
    angle = env.vehicle.get_transform().rotation.yaw - env.current_waypoint.transform.rotation.yaw
    speed_kmh = env.get_vehicle_lon_speed()
    speed_reward = calculate_speed_reward(speed_kmh)

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward

reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)
def calculate_traffic_light_reward(env):
    light_state = env._get_traffic_light_state()
    speed = env.get_vehicle_lon_speed()

    # Define penalties based on traffic light state
    if light_state == 0 and speed > 1:  # Red light and moving
        return -10  # Major penalty for moving on red light
    elif light_state == 1:  # Yellow light, cautious behavior encouraged
        return -5 if speed > 10 else 2
    elif light_state == 2:  # Green light
        return 5  # Reward for moving during green light
    return 0  # No reward or penalty if light is off or unknown
def reward_fn_traffic_aware(env):
    """Reward function that considers speed, traffic light, centering, and angle."""
    base_reward = reward_fn5(env)  #  reward_fn5 is basic driving reward function
    traffic_light_reward = calculate_traffic_light_reward(env)
    #print(traffic_light_reward)
    # Combine rewards
    reward = base_reward + traffic_light_reward
    return reward

reward_functions["reward_fn_traffic_aware"] = create_reward_fn(reward_fn_traffic_aware)

def combined_reward_function(env):
    """
    Combined reward function that integrates rewards based on:
    - Basic driving behavior (speed, angle, centering).
    - Compliance with traffic lights.
    - Progression through waypoints.
    """
    # Calculate the base driving reward from reward_fn5
    base_driving_reward = reward_fn5(env)
   # print(f"base reward:{base_driving_reward}")
    # Calculate the reward for correctly interacting with traffic lights
    traffic_light_reward = calculate_traffic_light_reward(env)
   # print(f"traffic light reward: {traffic_light_reward}")
    # Calculate the reward for navigating waypoints effectively
    waypoint_navigation_reward = reward_fn_waypoints(env)
    #print(f"waypoint reward: {waypoint_navigation_reward}")
    # Sum up all the rewards and penalties
    total_reward = base_driving_reward + traffic_light_reward + waypoint_navigation_reward

    return total_reward

# Now, create the reward function in the reward functions dictionary
reward_functions["combined_reward"] = create_reward_fn(combined_reward_function)

### additinial rewards #######################
def calculate_curvature_adapted_speed_reward(env):
    curvature = env.calculate_road_curvature()  # This would need to be implemented
    adjusted_target_speed = target_speed * (1 - min(curvature / max_curvature_threshold, 1))
    speed_kmh = env.get_vehicle_lon_speed()
    return calculate_speed_reward(speed_kmh, min_speed, adjusted_target_speed, max_speed)

def calculate_traffic_adherence_reward(env):
    # This function would need access to traffic light state and proximity
    # to stop signs within CARLA's API
    if env.is_at_traffic_light():
        if env.is_red_light():
            return penalty_reward  # Penalize if the car doesn't stop for red lights
    return 0  # No penalty if not at a traffic light or it's green

def calculate_near_collision_penalty(env):
    # Requires implementation of a function to detect near collisions
    if env.detect_near_collision():
        return penalty_reward
    return 0

def calculate_smoothness_penalty(env):
    # Requires storing previous actions to calculate the rate of change
    acceleration_change = np.abs(env.current_acceleration - env.previous_acceleration)
    steering_change = np.abs(env.current_steering - env.previous_steering)
    penalty = -np.exp(acceleration_change) - np.exp(steering_change)
    return penalty

def enhanced_reward_function(env):
    reward = 0
    reward += calculate_curvature_adapted_speed_reward(env)
    reward += calculate_traffic_adherence_reward(env)
    reward += calculate_near_collision_penalty(env)
    reward += calculate_smoothness_penalty(env)
    # Incorporate the base reward components
    reward += reward_fn5(env)  # Assuming reward_fn5 is your base reward function
    return reward
