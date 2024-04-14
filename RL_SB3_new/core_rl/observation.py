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
            "camera": Box(low=0.0, high=255.0, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "semantic_camera": Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
            "speed": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "acceleration": Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "distance_to_center": Box(low=np.array([0]), high=np.array([np.inf]), shape=(1,), dtype=np.float32),
            "angle_difference": Box(low=np.array([-np.pi]), high=np.array([np.pi]), shape=(1,), dtype=np.float32),
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