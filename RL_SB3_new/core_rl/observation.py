import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete

class CarlaObservations():

    def __init__(self, img_height, img_width):

        self.img_height = img_height
        self.img_width = img_width

    def get_observation_space(self):
        #return gym.spaces.Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        space = Dict({
            #"camera": Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "semantic_camera": Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "semantic_camera_left": Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "semantic_camera_right": Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "semantic_camera_back": Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "lidar_space" : Box(low=0.0, high=1.0, shape=(10000,), dtype=np.float32),
            "speed": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "acceleration": Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "distance_to_center": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "angle_difference": Box(low=np.array([-np.pi]), high=np.array([np.pi]), shape=(1,), dtype=np.float32),
            "traffic_light_state": Discrete(4), # 0: Red, 1: Yellow, 2: Green, 3: Off
            "yaw_rate": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "roll": Box(low=-180, high=180, shape=(1,), dtype=np.float32),
            "pitch": Box(low=-180, high=180, shape=(1,), dtype=np.float32),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),  # Assuming goal is a 3D location
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)   # Assuming goal is a 3D location
            })
        
        return space
# class EnhancedCarlaObservations(CarlaObservations):
#     def __init__(self, img_height, img_width):
#         super().__init__(img_height, img_width)
    
#     def get_observation_space(self):
#         space = Dict({
#             "camera": Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
#             "speed": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
#             "acceleration": Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
#             "distance_to_center": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
#             "angle_difference": Box(low=np.array([-np.pi]), high=np.array([np.pi]), shape=(1,), dtype=np.float32),
#             #"traffic_light_state": Discrete(3),  # Assuming 0: Green, 1: Yellow, 2: Red
#             #"proximity_to_nearest_vehicle": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
#             # Add more as needed
#         })
#         return space