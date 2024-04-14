import carla
import gym
import time
import random
import numpy as np
import math
from gym import spaces
from collections import deque
import pygame
import cv2
import xml.etree.ElementTree as ET


from utilities.graphics import HUD
from utilities.utils import get_actor_display_name, smooth_action, vector, distance_to_line, build_projection_matrix, get_image_point
from core_rl.actions import CarlaActions
from core_rl.observation import CarlaObservations
from utilities.planner import compute_route_waypoints
from utilities.utils import load_route_from_xml
# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, host, port, town, fps, obs_sensor, obs_res, view_res, reward_fn, action_smoothing, allow_render=True, allow_spectator=True, xml_file_path=None, route_id=None):
        
        self.obs_width, self.obs_height = obs_res
        self.spectator_width, self.spectator_height = view_res
        self.allow_render = allow_render
        self.allow_spectator = allow_spectator
        self.spectator_camera = None
        self.episode_idx = -2
        self.world = None
        self.fps = fps
        self.actions = CarlaActions()
        self.observations = CarlaObservations(self.obs_height, self.obs_width)
        self.obs_sensor = obs_sensor
        self.control = carla.VehicleControl()
        self.action_space = self.actions.get_action_space()
        self.observation_space = self.observations.get_observation_space()
        self.max_distance = 3000
        self.action_smoothing = action_smoothing
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.xml_file_path = xml_file_path
        self.route_id = route_id
        
        try:
            self.client = carla.Client(host, port)  
            self.client.set_timeout(100.0)

            self.client.load_world(map_name=town)
            self.world = self.client.get_world()
            self.world.set_weather(carla.WeatherParameters.ClearNoon)  
            self.world.apply_settings(
                carla.WorldSettings(  
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
            self.client.reload_world(False)  # reload map keeping the world settings
            self.map = self.world.get_map()
            
            # Spawn Vehicle
            self.tesla = self.world.get_blueprint_library().filter('model3')[0]
            self.start_transform = self._get_start_transform()
            self.curr_loc = self.start_transform.location
            self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)

           # Spawn collision and Lane invasion sensors
            colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
            lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
            self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
            self.colsensor.listen(self._collision_data)
            self.lanesensor.listen(self._lane_invasion_data)  

            # Create hud and initialize pygame for visualization
            if self.allow_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width, self.spectator_height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            # Set observation image
            if 'rgb' in self.obs_sensor:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            elif 'semantic' in self.obs_sensor:
                self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            else:
                raise NotImplementedError('unknown sensor type')

            self.rgb_cam.set_attribute('image_size_x', f'{self.obs_width}')
            self.rgb_cam.set_attribute('image_size_y', f'{self.obs_height}')
            self.rgb_cam.set_attribute('fov', '90')

            bound_x = self.vehicle.bounding_box.extent.x
            transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))
            self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
            self.sensor_front.listen(self._set_observation_image)          
 
            # Set spectator cam   
            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', f'{self.spectator_width}')
                self.spectator_camera.set_attribute('image_size_y', f'{self.spectator_height}')
                self.spectator_camera.set_attribute('fov', '100')
                transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-10.0))
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, transform, attach_to=self.vehicle)
                self.spectator_sensor.listen(self._set_viewer_image)
                    
            self.reset()
        except RuntimeError as msg:
            pass
    # Resets environment for new episode
    def reset(self):

        self.episode_idx += 1
        self.num_routes_completed = -1

        # Generate a random route
        self.generate_route(xml_file_path='routes/routes_town10all.xml', route_id='0')

        self.closed = False
        self.terminate = False
        self.success_state = False
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0

        # reset metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.world.tick()

        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)

        return obs
    
    # def generate_route(self):
    #     # Do a soft reset (teleport vehicle)
    #     self.control.steer = float(0.0)
    #     self.control.throttle = float(0.0)
    #     self.vehicle.set_simulate_physics(False)  # Reset the car's physics

    #     # Generate waypoints along the lap

    #     spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)
    #     route_length = 1
    #     while route_length <= 1:
    #         self.start_wp, self.end_wp = [self.map.get_waypoint(spawn.location) for spawn in
    #                                       spawn_points_list]
    #         self.route_waypoints = compute_route_waypoints(self.map, self.start_wp, self.end_wp, resolution=1.0)
    #         route_length = len(self.route_waypoints)
    #         if route_length <= 1:
    #             spawn_points_list = np.random.choice(self.map.get_spawn_points(), 2, replace=False)

    #     self.distance_from_center_history = deque(maxlen=30)

    #     self.current_waypoint_index = 0
    #     self.num_routes_completed += 1
    #     self.vehicle.set_transform(self.start_wp.transform)
    #     time.sleep(0.2)
    #     self.vehicle.set_simulate_physics(True)  
    def generate_route(self, xml_file_path, route_id='0'):
        # Assuming load_route_from_xml returns a list of carla.Location objects
        waypoints = load_route_from_xml(xml_file_path, route_id)

        # Convert carla.Location to carla.Waypoint
        start_location = waypoints[0]
        end_location = waypoints[-1]
        start_waypoint = self.map.get_waypoint(start_location)
        end_waypoint = self.map.get_waypoint(end_location)

        # Use compute_route_waypoints to generate the detailed route
        self.route_waypoints = compute_route_waypoints(self.map, start_waypoint, end_waypoint, resolution=1.0)

        # Setup the vehicle at the starting point
        if self.route_waypoints:
            start_wp, _ = self.route_waypoints[0]  # Assuming the first tuple contains the start waypoint
            self.vehicle.set_transform(start_wp.transform)
        
        # Initialize route tracking variables
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_simulate_physics(True)  # Re-enable physics for the vehicle
        self.distance_from_center_history = deque(maxlen=100)

    # Steps environment
    def step(self, action):
        # Apply action if provided, otherwise just tick the world
        if action is not None:
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                self.success_state = True
            # Unpack action into throttle and steer
            throttle, steer = [float(a) for a in action]
            # Smooth action application for throttle and steer
            self.control.throttle = smooth_action(self.control.throttle, throttle, self.action_smoothing)
            self.control.steer = smooth_action(self.control.steer, steer, self.action_smoothing)
            # Apply the control to the vehicle
            self.vehicle.apply_control(self.control)
        # Progress simulation by one step
        self.world.tick()
         # Gather observations
        self.observation = self._get_observation()
        if self.allow_spectator:
            self.viewer_image = self._get_viewer_image()
        # Update vehicle transform for new frame
        transform = self.vehicle.get_transform()
        # Update waypoint index
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            # Direct access to waypoint without unpacking
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                        vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1     # Passed the waypoint
            else:
                break        # Not yet passed the waypoint
        self.current_waypoint_index = waypoint_index

        # Handling the next and current waypoints without road maneuvers
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, _ = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
        else:
            self.next_waypoint = None  # Reached end of route
        # Current waypoint for this step
        self.current_waypoint, _ = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
         # Update completion metrics       
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(self.route_waypoints)
        # Calculate deviation from the center of the lane
        if self.next_waypoint:
            self.distance_from_center = distance_to_line(
                vector(self.current_waypoint.transform.location),
                vector(self.next_waypoint.transform.location),
                vector(transform.location))
        else:
            self.distance_from_center = 0  # or appropriate handling if next waypoint doesn't exist
        self.center_lane_deviation += self.distance_from_center
        # Update travel distance and speed accumulation
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        self.speed_accum += self.get_vehicle_lon_speed()
        # Check for termination conditions
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)
        
         # Update reward
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1
        
        # Render the environment if allowed
        if self.allow_render:
            self.render_events()

        # Compile information dictionary
        info = self.compile_info()

        return self.observation, self.last_reward, self.terminate or self.success_state, info
    def render_events(self):
        pygame.event.pump()
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            self.close()
            self.terminate = True
        self.render()

    def compile_info(self):
        return {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }
    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()

        self.closed = True
    
    def render(self, mode="human"):
 
        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)
  
        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.allow_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.spectator_camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation['camera'].shape[0], self.observation['camera'].shape[1]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.get_semantic_image(self.observation['camera']).swapaxes(0, 1)), pos_observation)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list
        # Render to screen
        pygame.display.flip()

    def get_vehicle_lon_speed(self):
        carla_velocity_vec3 = self.vehicle.get_velocity()
        vec4 = np.array([carla_velocity_vec3.x,
                         carla_velocity_vec3.y,
                         carla_velocity_vec3.z, 1]).reshape(4, 1)
        carla_trans = np.array(self.vehicle.get_transform().get_matrix())
        carla_trans.reshape(4, 4)
        carla_trans[0:3, 3] = 0.0
        vel_in_vehicle = np.linalg.inv(carla_trans) @ vec4
        return vel_in_vehicle[0]  

    def get_rgb_image(self, input):
        # Converting to suitable format for opencv function
        image = np.frombuffer(input.raw_data, dtype=np.uint8)
        image = image.reshape((input.height, input.width, 4))
        image = image[: ,: ,:3]
        image = image[:, :, ::-1].copy()

        return image

    def get_semantic_image(self, input):
        
        image = np.frombuffer(np.ascontiguousarray(input), dtype=np.uint8)
        image = image.reshape((input.shape[0], input.shape[1], 3))
        image = image[:, :, 2]
        classes = {
            0: [0, 0, 0],         # None
            1: [70, 70, 70],      # Buildings
            2: [190, 153, 153],   # Fences
            3: [72, 0, 90],       # Other
            4: [220, 20, 60],     # Pedestrians
            5: [153, 153, 153],   # Poles
            6: [157, 234, 50],    # RoadLines
            7: [128, 64, 128],    # Roads
            8: [244, 35, 232],    # Sidewalks
            9: [107, 142, 35],    # Vegetation
            10: [0, 0, 255],      # Vehicles
            11: [102, 102, 156],  # Walls
            12: [220, 220, 0]     # TrafficSigns
    }
        result = np.zeros((image.shape[0], image.shape[1], 3))
        for key, value in classes.items():
            result[np.where(image == key)] = value
        return result
        
    def _destroy_agents(self):

        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):

        # What we collided with and what was the impulse
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminate = True
        if self.allow_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

        #collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        
    def _lane_invasion_data(self, event):

        self.terminate = True
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.allow_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    # def _get_observation(self):
    #     while self.observation_buffer is None:
    #         pass
    #     obs = self.observation_buffer
    #     self.observation_buffer = None
    #     return obs

    def _get_observation(self):
        # Wait for the observation buffer to be filled
        while self.observation_buffer is None:
            time.sleep(0.01)  # Sleep to prevent CPU spinning, replace with a more appropriate waiting mechanism if possible

        # Retrieve and reset the buffer
        raw_image = self.observation_buffer
        self.observation_buffer = None

        # Convert raw image to a format suitable for the model (e.g., numpy array)
        processed_image = self.process_image(raw_image)  # Implement this method based on your image processing needs

        # Add additional sensor readings
        additional_obs = self.get_observations()

        # Combine image and additional sensor data into a single dictionary
        observation = {
            "camera": processed_image,  # Assuming process_image returns a numpy array or similar structure
            'semantic_camera': self.get_semantic_image(self.process_image(raw_image)),
            **additional_obs  # Merge additional observations into the main observation dictionary
        }

        return observation
    def process_image(self, image):
        # Example processing, adjust according to your needs
        return np.array(image.raw_data).reshape((self.obs_height, self.obs_width, 4))[:, :, :3]  # Assuming BGRA to BGR conversion


    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer
        self.viewer_image_buffer = None
        return image

    def _get_start_transform(self):
        return random.choice(self.map.get_spawn_points())  

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(image.transform.get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(image.height)
        image_h = int(image.width)
        fov = float(image.fov)

        image = self.get_rgb_image(image)

        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_h, image_w, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image    

    def get_speed(self):
        # Assuming 'self.vehicle' is your carla.VehicleActor instance
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Speed in m/s
        return speed_m_s * 3.6  # Convert m/s to km/h

    def get_acceleration(self):
        # Retrieve acceleration from CARLA (if available) or calculate it
        acceleration = self.vehicle.get_acceleration()
        acceleration_magnitude = np.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
        return acceleration_magnitude  # In m/s^2
    def get_distance_to_center(self):
        # Get the vehicle's location
        location = self.vehicle.get_location()
        # Get the nearest waypoint in the center of the lane
        waypoint = self.map.get_waypoint(location)
        # Calculate the distance from the waypoint (center of the lane) to the vehicle
        # This might require additional logic to ensure it's the lateral distance
        distance = location.distance(waypoint.transform.location)
        return distance
    def get_angle_difference(self):
        # Get vehicle's orientation
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw = vehicle_transform.rotation.yaw
        
        # Get the direction of the road from waypoints
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        road_yaw = waypoint.transform.rotation.yaw
        
        # Calculate the difference
        angle_difference = np.deg2rad(vehicle_yaw - road_yaw)
        # Normalize the angle to the range [-pi, pi]
        angle_difference = np.arctan2(np.sin(angle_difference), np.cos(angle_difference))
        
        return angle_difference
    def get_observations(self):
        # Collect all relevant data for the observation space
        
        speed = self.get_speed()
        acceleration = self.get_acceleration()
        distance_to_center = self.get_distance_to_center()
        angle_difference = self.get_angle_difference()

        # Package into a dictionary exactly matching the defined observation_space
        observations = {
            'speed': np.array([speed], dtype=np.float32),
            'acceleration': np.array([acceleration], dtype=np.float32),
            'distance_to_center': np.array([distance_to_center], dtype=np.float32),
            'angle_difference': np.array([angle_difference], dtype=np.float32)
        }
        return observations


    def generate_dense_route(self, sparse_waypoints):
        dense_route = []
        for i in range(len(sparse_waypoints) - 1):
            # Get the starting waypoint from the map
            start_wp = self.map.get_waypoint(sparse_waypoints[i])
            end_wp = self.map.get_waypoint(sparse_waypoints[i + 1])
            
            # This is a simplified approach to add waypoints. You might need to adjust based on your map and CARLA version.
            # For a more sophisticated approach, consider using CARLA's routing functionalities or a custom pathfinding algorithm.
            dense_route.append(start_wp)
            next_wps = start_wp.next_until_lane_end(2.0)  # Generate waypoints every 2 meters until the lane end
            dense_route.extend(next_wps)
            
            # Ensure the last waypoint is added to the route
            if end_wp not in dense_route:
                dense_route.append(end_wp)
        
        return dense_route 
    print("finished")        
    
        