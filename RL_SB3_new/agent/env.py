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

from navigation.basic_agent import BasicAgent
from utilities.graphics import HUD
from utilities.utils import get_actor_display_name, smooth_action, vector, distance_to_line, build_projection_matrix, get_image_point
from core_rl.actions import CarlaActions
from core_rl.observation import CarlaObservations
from utilities.planner import compute_route_waypoints, generate_route
from utilities.utils import load_route_from_xmlnew, get_all_route_ids, load_route_from_xmlold, draw_route
from utilities.visualize_multiple_sensors import SensorManager, DisplayManager
from agent.rewards import combined_reward_function, reward_fn5, calculate_traffic_light_reward, reward_fn_waypoints
# Carla environment
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
OFF_COLOR = (200, 200, 200)  # Light grey to indicate 'Off' or no traffic light
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, host, port, town, fps, obs_sensor_semantic, obs_sensor_rgb, obs_res, view_res, compute_reward, action_smoothing, allow_render=True, allow_spectator=True, xml_file_path=None, route_id=None):
        
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
        #self.obs_sensor = obs_sensor
        self.obs_sensor_semantic = obs_sensor_semantic
        self.obs_sensor_rgb = obs_sensor_rgb
        self.control = carla.VehicleControl()
        self.action_space = self.actions.get_action_space()
        self.observation_space = self.observations.get_observation_space()
        self.max_distance = 3000
        self.action_smoothing = action_smoothing
        #self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.compute_reward = (lambda x: 0) if not callable(compute_reward) else compute_reward
        self.xml_file_path = xml_file_path
        self.route_id = route_id
        self.spawned_vehicles = []
        
        
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
            #self.vehicle = self.world.spawn_actor(self.tesla, self.start_transform)
            self.vehicle = self.world.try_spawn_actor(self.tesla, self.start_transform)
            if not self.vehicle:
                raise Exception("Failed to spawn ego vehicle.")
            self.initialize_traffic_manager()
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
                #self.display = pygame.display.set_mode((self.spectator_width, self.spectator_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.display_manager = DisplayManager(grid_size=[2.4, 3.6], window_size=[self.spectator_width, self.spectator_height])
                self.clock = pygame.time.Clock()
                self.hud = HUD(self.spectator_width*0.6, self.spectator_height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)
                self.setup_sensors()
            
                # Set observation image
                # if 'rgb' in self.obs_sensor_rgb:
                #     self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
                # else:
                #     raise NotImplementedError('unknown sensor type')
                
                # self.rgb_cam.set_attribute('image_size_x', f'{self.obs_width}')
                # self.rgb_cam.set_attribute('image_size_y', f'{self.obs_height}')
                # self.rgb_cam.set_attribute('fov', '90')

                # bound_x = self.vehicle.bounding_box.extent.x
                # transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4), carla.Rotation(yaw=0))
                # self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
                # self.sensor_front.listen(self._set_observation_rgb)

                # if 'rgb' in self.obs_sensor_rgb:
                #     self.rgb_cam_left = self.world.get_blueprint_library().find('sensor.camera.rgb')
                # else:
                #     raise NotImplementedError('unknown sensor type')
                
                # self.rgb_cam_left.set_attribute('image_size_x', f'{self.obs_width}')
                # self.rgb_cam_left.set_attribute('image_size_y', f'{self.obs_height}')
                # self.rgb_cam_left.set_attribute('fov', '90')

                # bound_x = self.vehicle.bounding_box.extent.x
                # transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4), carla.Rotation(yaw=-90))
                # self.sensor_left = self.world.spawn_actor(self.rgb_cam_left, transform_front, attach_to=self.vehicle)
                # self.sensor_left.listen(self._set_observation_rgb_left)

                # if 'rgb' in self.obs_sensor_rgb:
                #     self.rgb_cam_right = self.world.get_blueprint_library().find('sensor.camera.rgb')
                # else:
                #     raise NotImplementedError('unknown sensor type')
                
                # self.rgb_cam_right.set_attribute('image_size_x', f'{self.obs_width}')
                # self.rgb_cam_right.set_attribute('image_size_y', f'{self.obs_height}')
                # self.rgb_cam_right.set_attribute('fov', '90')

                # bound_x = self.vehicle.bounding_box.extent.x
                # transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4), carla.Rotation(yaw=90))
                # self.sensor_right = self.world.spawn_actor(self.rgb_cam_right, transform_front, attach_to=self.vehicle)
                # self.sensor_right.listen(self._set_observation_rgb_right)

                if 'semantic' in self.obs_sensor_semantic:
                    self.semantic_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                else:
                    raise NotImplementedError('unknown sensor type')
                self.semantic_cam.set_attribute('image_size_x', f'{self.obs_width}')
                self.semantic_cam.set_attribute('image_size_y', f'{self.obs_height}')
                self.semantic_cam.set_attribute('fov', '90')  

                bound_x = self.vehicle.bounding_box.extent.x
                transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4))
                self.sensor_front1 = self.world.spawn_actor(self.semantic_cam, transform_front, attach_to=self.vehicle)
                self.sensor_front1.listen(self._set_observation_semantic)  

                if 'semantic' in self.obs_sensor_semantic:
                    self.semantic_cam_left = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                else:
                    raise NotImplementedError('unknown sensor type')
                self.semantic_cam_left.set_attribute('image_size_x', f'{self.obs_width}')
                self.semantic_cam_left.set_attribute('image_size_y', f'{self.obs_height}')
                self.semantic_cam_left.set_attribute('fov', '90')  

               
                transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4), carla.Rotation(yaw=-90))
                self.sensor_left = self.world.spawn_actor(self.semantic_cam_left, transform_front, attach_to=self.vehicle)
                self.sensor_left.listen(self._set_observation_semantic_left)  

                if 'semantic' in self.obs_sensor_semantic:
                    self.semantic_cam_right = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                else:
                    raise NotImplementedError('unknown sensor type')
                self.semantic_cam_right.set_attribute('image_size_x', f'{self.obs_width}')
                self.semantic_cam_right.set_attribute('image_size_y', f'{self.obs_height}')
                self.semantic_cam_right.set_attribute('fov', '90')  

               
                transform_front = carla.Transform(carla.Location(x=bound_x, z=2.4), carla.Rotation(yaw=90))
                self.sensor_right = self.world.spawn_actor(self.semantic_cam_right, transform_front, attach_to=self.vehicle)
                self.sensor_right.listen(self._set_observation_semantic_right)  

                if 'semantic' in self.obs_sensor_semantic:
                    self.semantic_cam_back = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
                else:
                    raise NotImplementedError('unknown sensor type')
                self.semantic_cam_back.set_attribute('image_size_x', f'{self.obs_width*2}')
                self.semantic_cam_back.set_attribute('image_size_y', f'{self.obs_height}')
                self.semantic_cam_back.set_attribute('fov', '90')  

               
                transform_front = carla.Transform(carla.Location(x=-1, z=2.4), carla.Rotation(yaw=180))
                self.sensor_back = self.world.spawn_actor(self.semantic_cam_right, transform_front, attach_to=self.vehicle)
                self.sensor_back.listen(self._set_observation_semantic_back)  

            # Set spectator cam   
            if self.allow_spectator:
                self.spectator_camera = self.world.get_blueprint_library().find('sensor.camera.rgb')
                self.spectator_camera.set_attribute('image_size_x', f'{self.spectator_width}')
                self.spectator_camera.set_attribute('image_size_y', f'{self.spectator_height}')
                self.spectator_camera.set_attribute('fov', '90')
                transform = carla.Transform(carla.Location(x=-10, z=10), carla.Rotation(pitch=-45))
                self.spectator_sensor = self.world.spawn_actor(self.spectator_camera, transform, attach_to=self.vehicle)
                self.spectator_sensor.listen(self._set_viewer_image)
                        # Example setup for a single RGB camera sensor
            if self.allow_spectator:
                

                # LIDAR
                lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('range', '100')
                lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
                lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
                lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])


                # Adjust the sensor location and rotation as needed
                lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
                self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
                self.lidar_sensor.listen(self.handle_lidar_data)

            self.reset()
        except RuntimeError as msg:
            pass

    def setup_sensors(self):
                camera_options = {'fov': '90'}
                camera_transform_left = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90))
                self.rgb_camera_left = SensorManager(self.world, self.display_manager, 'RGBCamera', camera_transform_left, self.vehicle, {}, display_pos=[0, 0.66])
                camera_transform_center = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00))
                self.rgb_camera_center = SensorManager(self.world, self.display_manager, 'RGBCamera', camera_transform_center, self.vehicle, {}, display_pos=[0, 1.66])
                camera_transform_right = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90))
                self.rgb_camera_right = SensorManager(self.world, self.display_manager, 'RGBCamera', camera_transform_right, self.vehicle, {}, display_pos=[0, 2.66])
                camera_transform_back = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+180))
                self.rgb_camera_back = SensorManager(self.world, self.display_manager, 'RGBCamera', camera_transform_back, self.vehicle, {}, display_pos=[1, 1.66])
                lidar_position = carla.Transform(carla.Location(x=0, z=2.4))
                self.Lidar = SensorManager(self.world, self.display_manager, 'LiDAR', lidar_position, self.vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=[1, 2.66])
                #self.Lidar = SensorManager(self.world, self.display_manager, 'SemanticLiDAR', lidar_position, self.vehicle, {'channels' : '64', 'range' : '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=[1, 2])
                # camera_transform_top = carla.Transform(carla.Location(x=0, z=20), carla.Rotation(pitch=-90))
                # self.topdown_camera = SensorManager(self.world, self.display_manager, 'RGBCamera', camera_transform_top, self.vehicle, {}, display_pos=[1, 0.9])
                #self.topdown_camera.sensor.listen(self._set_observation_semantic)  
    # Resets environment for new episode
    def reset(self):

        self.episode_idx += 1
        self.num_routes_completed = -1
        

        # Generate a random route
        generate_route(self, xml_file_path='RL_SB3_new/routes/dongfeng.xml')
        self.cleanup()
        
        # ego_vehicle_spawn_index = 28
        ego_vehicle_route_indices = [28, 124, 30, 31, 33, 97, 107, 58, 131, 65, 63, 152, 148, 74, 153, 145, 135, 139, 110, 102, 116, 27, 122, 25, 78, 80, 91, 0, 103, 111, 95, 141, 109, 14, 12, 125, 7, 9]
        # self.vehicle = self.spawn_and_assign_routes([ego_vehicle_spawn_index], ego_vehicle_route_indices)[0]
        other_vehicle_spawn_indices = [28, 130, 29, 70, 73] # 17, 18, 97, 107, 64, 128, 79, 137, 131
        self.spawned_vehicles = self.spawn_and_assign_routes(other_vehicle_spawn_indices, ego_vehicle_route_indices)
        
        self.closed = False
        self.terminate = False
        self.success_state = False
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = []
        # self.observation_rgb = self.observation_buffer_rgb = None  # Last rgb received observation
        # self.observation_rgb_left = self.observation_buffer_rgb_left = None
        # self.observation_rgb_right = self.observation_buffer_rgb_right = None
        self.observation_semantic = self.observation_buffer_semantic = None # Last semantic received observation
        self.observation_semantic_left = self.observation_buffer_semantic_left = None
        self.observation_semantic_right = self.observation_buffer_semantic_right = None
        self.observation_semantic_right = self.observation_buffer_semantic_back = None
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0

        # reset metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.base_driving_reward = 0
        self.traffic_light_reward = 0
        self.waypoint_navigation_reward = 0
        self.world.tick()

        # Retrieve traffic light state and convert to human-readable format
        self.base_driving_reward=0
        self.traffic_light_reward=0
        self.waypoint_navigation_reward=0
        traffic_light_state_int = self._get_traffic_light_state()
        traffic_light_states = {0: "Red", 1: "Yellow", 2: "Green", 3: "Off"}
        self.traffic_light_state_str = traffic_light_states.get(traffic_light_state_int, "Unknown")
        
        
        # self.base_reward = reward_fn5(self)
        # self.traffic_reward = calculate_traffic_light_reward(self)
        # self.waypoint_reward = reward_fn_waypoints(self)
        
        # Return initial observation
        time.sleep(0.2)
        obs = self.step(None)[0]
        time.sleep(0.2)
      
        return obs
     
    

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
        self.observation = self.get_observation()
        if self.allow_spectator:
            self.viewer_image = self._get_viewer_image()
            # self.observation_rgb = self._get_observation_rgb()
            # self.observation_rgb_left = self._get_observation_rgb_left()
            # self.observation_semantic = self._get_observation_semantic()
            # self.observation_semantic_left = self._get_observation_rgb_left()
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
        self.base_driving_reward=reward_fn5(self)
        self.traffic_light_reward=calculate_traffic_light_reward(self)
        self.waypoint_navigation_reward=reward_fn_waypoints(self)
        # print(f"reward_base {reward_fn5(self)}")
        # print(f"reward_traffic {calculate_traffic_light_reward(self)}")
        # print(f"reward_waypoint {reward_fn_waypoints(self)}")
        self.last_reward = self.compute_reward(self)
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
            "Driving Reward:      % 7.2f" % self.base_driving_reward,
            "Traffic Light Reward:% 7.2f" % self.traffic_light_reward,
            "Waypoint Reward:     % 7.2f" % self.waypoint_navigation_reward,
            "Combined Reward:     % 7.2f" % self.last_reward,
            "Total reward:        % 7.2f" % self.total_reward,
            "",
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Traffic light state: {}".format(self.traffic_light_state_str)
        ])
        if self.allow_spectator:
            # Blit image from spectator camera

            
            #self.draw = self.draw_route(self.display_manager.sensor_list[0].surface, self.display_manager.sensor_list[0].world.get_transform(), self.route_waypoints, (self.spectator_width, self.spectator_height))
            self.display_manager.render()
            #self.topdown_camera.render(self.display_manager)
            
            #camera_transform= self.rgb_camera_center.get_sensor().get_transform()
            # top_down_display_surface = self.display_manager.sensor_list[0].surface
            # top_down_display_surface.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            # Draw the route on the pygame window
            #draw_route(self.display_manager.display, self.route_waypoints, camera_transform, (self.display_manager.window_size[0], self.display_manager.window_size[1]))
            
            # Traffic light
            traffic_light_state = self._get_traffic_light_state()
            traffic_light_color = OFF_COLOR  # Default to 'Off' color
        
            if traffic_light_state == 0:  # Red
                traffic_light_color = RED
            elif traffic_light_state == 1:  # Yellow
                traffic_light_color = YELLOW
            elif traffic_light_state == 2:  # Green
                traffic_light_color = GREEN
         # Position for the traffic light indicator on the screen
            traffic_light_position = (self.display_manager.window_size[0] - 1100, 550)
            pygame.draw.circle(self.display_manager.display, traffic_light_color, traffic_light_position, 20)

        # Superimpose current observation into top-right corner
            obs_h, obs_w = self.observation['semantic_camera'].shape[0], self.observation['semantic_camera'].shape[1]
            
            
            # Blit semantic camera image
            semantic_position_front = (self.display_manager.window_size[0] - int(4*obs_w), self.display_manager.window_size[1] - int(obs_h))
            semantic_image_surface = pygame.surfarray.make_surface((self.observation['semantic_camera']).swapaxes(0, 1))
            self.display_manager.display.blit(semantic_image_surface, semantic_position_front)

            semantic_position_left = (self.display_manager.window_size[0] - int(5*obs_w), self.display_manager.window_size[1] - int(obs_h))
            semantic_image_surface_left = pygame.surfarray.make_surface((self.observation['semantic_camera_left']).swapaxes(0, 1))
            self.display_manager.display.blit(semantic_image_surface_left, semantic_position_left)

            semantic_position_right = (self.display_manager.window_size[0] - int(3*obs_w), self.display_manager.window_size[1] - int(obs_h))
            semantic_image_surface_right = pygame.surfarray.make_surface((self.observation['semantic_camera_right']).swapaxes(0, 1))
            self.display_manager.display.blit(semantic_image_surface_right, semantic_position_right)

            semantic_position_back = (self.display_manager.window_size[0] - int(2*obs_w), self.display_manager.window_size[1] - int(obs_h))
            semantic_image_surface_back = pygame.surfarray.make_surface((self.observation['semantic_camera_back']).swapaxes(0, 1))
            self.display_manager.display.blit(semantic_image_surface_back, semantic_position_back)
        # Blit RGB camera image
            # camera_position_front = (self.display_manager.window_size[0] - int(2*obs_w) - 10, 300)
            # camera_image_surface = pygame.surfarray.make_surface(self.observation['camera'].swapaxes(0, 1))
            # self.display_manager.display.blit(camera_image_surface, camera_position_front)

            # camera_position_left = (self.display_manager.window_size[0] - int(3*obs_w) - 10, 300)
            # camera_image_surface_left = pygame.surfarray.make_surface(self.observation['camera_left'].swapaxes(0, 1))
            # self.display_manager.display.blit(camera_image_surface_left, camera_position_left)

            # camera_position_right = (self.display_manager.window_size[0] - int(obs_w) - 10, 300)
            # camera_image_surface_right = pygame.surfarray.make_surface(self.observation['camera_right'].swapaxes(0, 1))
            # self.display_manager.display.blit(camera_image_surface_right, camera_position_right)
        #Blit BIV camera image
            top_down_camera_position = (int(self.display_manager.window_size[0] - int(5*obs_w)), self.display_manager.window_size[1] - int(3.5*obs_h))
            self.viewer_image1 = self._draw_path(self.spectator_sensor, self.viewer_image)
            top_down_display_surface=(pygame.surfarray.make_surface(self.viewer_image1.swapaxes(0, 1)))
            new_width = int(top_down_display_surface.get_width()*0.40)
            new_height = int(top_down_display_surface.get_height()*0.45)
            new_size = (new_width, new_height)
            scaled_surface = pygame.transform.scale(top_down_display_surface, new_size)
            #             # Example: Crop the central part of the scaled surface
            crop_x = int(new_width * 0.15)  # Start cropping at 25% of the new width
            crop_y = int(new_height * 0.00)  # Start cropping at 25% of the new height
            crop_width = int(new_width * 0.69)  # Crop width is 50% of the new width
            crop_height = int(new_height * 0.93)  # Crop height is 50% of the new height
            crop_rect = pygame.Rect(crop_x, crop_y, crop_width, crop_height)

            # Create a subsurface (this does not create a new copy, it references the scaled surface)
            cropped_surface = scaled_surface.subsurface(crop_rect)
            self.display_manager.display.blit(cropped_surface, top_down_camera_position)
            ###
            lidar_raw = self.get_lidar_observation()
            lidar_surface = self.lidar_grid_to_surface(lidar_raw)
            
            self.display_manager.display.blit(lidar_surface, (self.display_manager.window_size[0] - int(obs_w), self.display_manager.window_size[1] - int(obs_h)))
        # Render HUD
        #self.hud.render(self.display, extra_info=self.extra_info)
        self.hud.render(self.display_manager.display, extra_info=self.extra_info)
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
        
        image = np.frombuffer(input.raw_data, dtype=np.uint8)
        image = image.reshape((input.height, input.width, 4))
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
        # if self.allow_render:
        #     self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

        #collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

    def _lane_invasion_data(self, event):
        # Set of lane marking types considered as terminable conditions
        terminable_lane_markings = {
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.SolidSolid,
            carla.LaneMarkingType.SolidBroken,
            carla.LaneMarkingType.BrokenSolid,
            # ... include any other lane types that should cause termination
        }

        # Determine the types of markings that were crossed
        crossed_markings = set(x.type for x in event.crossed_lane_markings)

        # Check if any of the crossed markings are considered terminable
        should_terminate = any(mark_type in terminable_lane_markings for mark_type in crossed_markings)
        
        if should_terminate:
            # Terminate if any of the crossed markings are terminable types
            self.terminate = False
        #     text = " and ".join([str(x).split('.')[-1] for x in crossed_markings if x in terminable_lane_markings])
        #     if self.allow_render:
        #         self.hud.notification(f"Crossed a terminable lane marking {text}")
        # else:
        #     # Otherwise, do not terminate
        #     if self.allow_render:
        #         self.hud.notification("Crossed a broken lane marking without termination")

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer
        self.viewer_image_buffer = None
        return image
    def _get_observation_rgb(self):
        while self.observation_buffer_rgb is None:
            pass
        image = self.observation_buffer_rgb
        self.observation_buffer_rgb = None
        return image
    # def _get_observation_rgb_left(self):
    #     while self.observation_buffer_rgb_left is None:
    #         pass
    #     image = self.observation_buffer_rgb_left
    #     self.observation_buffer_rgb_left = None
    #     return image
    # def _get_observation_rgb_right(self):
    #     while self.observation_buffer_rgb_right is None:
    #         pass
    #     image = self.observation_buffer_rgb_right
    #     self.observation_buffer_rgb_right = None
    #     return image
    def _get_observation_semantic(self):
        while self.observation_buffer_semantic is None:
            pass
        image = self.observation_buffer_semantic
        self.observation_buffer_semantic = None
        return image
    def _get_observation_semantic_left(self):
        while self.observation_buffer_semantic_left is None:
            pass
        image = self.observation_buffer_semantic_left
        self.observation_buffer_semantic_left = None
        return image
    def _get_observation_semantic_right(self):
        while self.observation_buffer_semantic_right is None:
            pass
        image = self.observation_buffer_semantic_right
        self.observation_buffer_semantic_right = None
        return image
    def _get_observation_semantic_back(self):
        while self.observation_buffer_semantic_back is None:
            pass
        image = self.observation_buffer_semantic_back
        self.observation_buffer_semantic_back = None
        return image
        
    def _set_observation_image(self, image):
        self.observation_buffer = image

    # def _set_observation_rgb(self, image):
    #     self.observation_buffer_rgb = image

    # def _set_observation_rgb_left(self, image):
    #     self.observation_buffer_rgb_left = image

    # def _set_observation_rgb_right(self, image):
    #     self.observation_buffer_rgb_right = image

    def _set_observation_semantic(self, image):
        self.observation_buffer_semantic = image

    def _set_observation_semantic_left(self, image):
        self.observation_buffer_semantic_left = image

    def _set_observation_semantic_right(self, image):
        self.observation_buffer_semantic_right = image

    def _set_observation_semantic_back(self, image):
        self.observation_buffer_semantic_back = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _draw_path(self, camera, image):
        """Draw a connected path from start of route to end using homography."""
        try:
            vehicle_vector = vector(self.vehicle.get_transform().location)
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            image_w = int(camera.attributes['image_size_x'])
            image_h = int(camera.attributes['image_size_y'])
            fov = float(camera.attributes['fov'])

            # Convert CARLA image to a numpy array
            np_image = self.get_rgb_image(image)

            for i, waypoint in enumerate(self.route_waypoints[self.current_waypoint_index:]):
                waypoint_location = waypoint[0].transform.location + carla.Location(z=1.25)
                waypoint_vector = vector(waypoint_location)
                if not (2 < np.linalg.norm(vehicle_vector - waypoint_vector) < 50):
                    continue

                K = build_projection_matrix(image_w, image_h, fov)
                x, y = get_image_point(waypoint_location, K, world_2_camera)

                # Validate if the point is within the image boundary before drawing
                if 0 <= x < image_w and 0 <= y < image_h:
                    color = (255, 0, 0) if i == len(self.route_waypoints) - 1 else (0, 0, 255)
                    np_image = cv2.circle(np_image, (int(x), int(y)), radius=3, color=color, thickness=-1)
                else:
                    None #print(f"Point out of bounds: (x: {x}, y: {y})")

        except Exception as e:
            print("Error in _draw_path:", str(e))

        return np_image



    def _get_speed(self):
        # Assuming 'self.vehicle' is your carla.VehicleActor instance
        velocity = self.vehicle.get_velocity()
        speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Speed in m/s
        return speed_m_s * 3.6  # Convert m/s to km/h

    def _get_acceleration(self):
        # Retrieve acceleration from CARLA (if available) or calculate it
        acceleration = self.vehicle.get_acceleration()
        acceleration_magnitude = np.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
        return acceleration_magnitude  # In m/s^2
    def _get_distance_to_center(self):
        # Get the vehicle's location
        location = self.vehicle.get_location()
        # Get the nearest waypoint in the center of the lane
        waypoint = self.map.get_waypoint(location)
        # Calculate the distance from the waypoint (center of the lane) to the vehicle
        # This might require additional logic to ensure it's the lateral distance
        distance = location.distance(waypoint.transform.location)
        return distance
    def _get_angle_difference(self):
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
    def process_image(self, image):
        if not image.raw_data:
            print("No image data received")
            return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)  # Return a black image
        else:
            image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
            if image_data.shape[0] != self.obs_height * self.obs_width * 4:
                print(f"Unexpected image data size: {image_data.shape[0]}")
                return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)  # Return a black image
            return image_data.reshape((self.obs_height, self.obs_width, 4))[:, :, :3]  # Assuming BGRA to BGR conversion

    def _get_traffic_light_state(self, threshold=10):
        vehicle_location = self.vehicle.get_location()
        lights = self.world.get_actors().filter('*traffic_light*')
        closest_light = None
        min_distance = float('inf')

        for light in lights:
            distance = vehicle_location.distance(light.get_location())
            if distance < min_distance:
                min_distance = distance
                closest_light = light

        if closest_light and min_distance <= threshold:
            state = closest_light.get_state()
            state_to_int = {
                carla.TrafficLightState.Red: 0,
                carla.TrafficLightState.Yellow: 1,
                carla.TrafficLightState.Green: 2,
                carla.TrafficLightState.Off: 3
            }
            return state_to_int.get(state, 3)
        else:
            return 3  # No relevant traffic light or too far
    def list_all_traffic_lights(self):
        # List all traffic lights to confirm they are loaded in the simulation
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        if not traffic_lights:
            print("No traffic lights are present in the simulation.")
        else:
            for light in traffic_lights:
                print(f"Traffic Light ID: {light.id}, Location: {light.get_location()}")
                
    def activate_traffic_lights(self):
        # Ensure that traffic lights are not frozen and are in a normal operational state
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        for light in traffic_lights:
            light.freeze(False)
            print(f"Activated Traffic Light ID: {light.id}")
    def find_nearest_traffic_light(self):
        vehicle_location = self.vehicle.get_location()
        traffic_lights = self.world.get_actors().filter('traffic_light')
        closest_light = None
        min_distance = float('inf')

        for light in traffic_lights:
            distance = vehicle_location.distance(light.get_location())
            if distance < min_distance and distance < 1000:  # Example: 1000 meters as a maximum search radius
                min_distance = distance
                closest_light = light

        if closest_light:
            print(f"Closest Traffic Light ID: {closest_light.id} at distance {min_distance}")
        else:
            print("No nearby traffic lights found within the search radius.")
    
    def handle_lidar_data(self, lidar_data):
        self.lidar_points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)[:, :3]  # Extracting x, y, z
        # print("Lidar points captured:", self.lidar_points.shape)  # Debugging line

    def get_lidar_observation(self):
        if not hasattr(self, 'lidar_points') or self.lidar_points.size == 0:
            print("No LiDAR data available.")
            return np.zeros((100, 100))  # Return an empty grid if no data

        grid_size = 100
        lidar_range = 50
        grid = np.zeros((grid_size, grid_size))

        scale = grid_size / (2 * lidar_range)
        for point in self.lidar_points:
            x, y, _ = point
            if -lidar_range <= x <= lidar_range and -lidar_range <= y <= lidar_range:
                grid_x = int((x + lidar_range) * scale)
                grid_y = int((y + lidar_range) * scale)
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_x, grid_y] += 1
                else:
                    print("Point out of grid bounds:", x, y, grid_x, grid_y)

        grid /= grid.max() if grid.max() != 0 else 1  # Avoid division by zero
        #print("LiDAR grid generated with max value:", grid.max())
        return grid

    def lidar_grid_to_surface(self, grid):
        # Normalize the grid to the range [0, 255]
        normalized_grid = (grid / grid.max() * 255).astype(np.uint8)
        
        # Stack the grid three times to create an RGB image
        rgb_grid = np.stack([normalized_grid]*3, axis=-1)
        
        # Create a Pygame surface from the RGB data
        unscaled_surface = pygame.surfarray.make_surface(rgb_grid)

        # Define the new size
        new_size = (196, 100)  # width=196, height=100

        # Scale the surface to the new size
        scaled_surface = pygame.transform.scale(unscaled_surface, new_size)

        return scaled_surface

    def get_yaw_rate(self):
        angular_velocity = self.vehicle.get_angular_velocity()
        return angular_velocity.z  # Assuming z is the yaw axis in your vehicle model

    def get_roll_pitch(self):
        rotation = self.vehicle.get_transform().rotation
        return rotation.roll, rotation.pitch
    def _get_achieved_goal(self):
        if self.current_waypoint_index > 0:
            loc = self.route_waypoints[self.current_waypoint_index - 1][0].transform.location
            #print(loc)
            return np.array([loc.x, loc.y, loc.z], dtype=np.float32)
        loc = self.route_waypoints[0][0].transform.location
        return np.array([loc.x, loc.y, loc.z], dtype=np.float32)

    def _get_desired_goal(self):
        loc = self.route_waypoints[-1][0].transform.location
        #print(loc)
        return np.array([loc.x, loc.y, loc.z], dtype=np.float32)

    def get_observation(self):
        # Wait for the observation buffer to be filled
        while self.observation_buffer_semantic is None or self.observation_buffer_semantic_left is None or self.observation_buffer_semantic_right is None or self.observation_buffer_semantic_back is None:
            time.sleep(0.01)  # Sleep to prevent CPU spinning, replace with more appropriate waiting mechanism if available
        # while self.observation_buffer_rgb_left is None:
        #     time.sleep(0.01)
        # while self.observation_buffer_rgb_right is None:
        #     time.sleep(0.01)
        # Retrieve and reset the buffer
        # raw_image = self.observation_buffer_rgb
        # self.observation_buffer_rgb = None

        # raw_image_left = self.observation_buffer_rgb_left
        # self.observation_buffer_rgb_left = None

        # raw_image_right = self.observation_buffer_rgb_right
        # self.observation_buffer_rgb_right = None

        raw_semantic =self.observation_buffer_semantic
        self.observation_buffer_semantic = None

        raw_semantic_left = self.observation_buffer_semantic_left
        self.observation_buffer_semantic_left = None

        raw_semantic_right = self.observation_buffer_semantic_right
        self.observation_buffer_semantic_right = None

        raw_semantic_back = self.observation_buffer_semantic_back
        self.observation_buffer_semantic_back = None
        # print(raw_image)
        # print(raw_image_left)
        # print(raw_image_right)


        # Process the raw image to a format suitable for the model (e.g., numpy array)
        # processed_image = self.process_image(raw_image)
        # processed_image_left = self.process_image(raw_image_left)
        # processed_image_right = self.process_image(raw_image_right)
        semantic_image = self.get_semantic_image(raw_semantic)
        semantic_image_left = self.get_semantic_image(raw_semantic_left)
        semantic_image_right = self.get_semantic_image(raw_semantic_right)
        semantic_image_back = self.get_semantic_image(raw_semantic_back)
        # Collect all additional sensor data
        lidar_obs = self.get_lidar_observation().flatten()
        speed = self._get_speed()
        acceleration = self._get_acceleration()
        distance_to_center = self._get_distance_to_center()
        angle_difference = self._get_angle_difference()
        traffic_light_state = self._get_traffic_light_state()
        yaw_rate = self.get_yaw_rate()
        roll, pitch = self.get_roll_pitch()
        achieved_goal = self._get_achieved_goal()
        desired_goal = self._get_desired_goal()
        # Package into a dictionary matching the defined observation_space
        observation = {
            # 'camera': processed_image,
            # 'camera_left': processed_image_left,
            # 'camera_right': processed_image_right,
            'semantic_camera': semantic_image,
            'semantic_camera_left': semantic_image_left,
            'semantic_camera_right': semantic_image_right,
            'semantic_camera_back': semantic_image_back,
            'lidar_space': lidar_obs,
            'speed': np.array([speed], dtype=np.float32),
            'acceleration': np.array([acceleration], dtype=np.float32),
            'distance_to_center': np.array([distance_to_center], dtype=np.float32),
            'angle_difference': np.array([angle_difference], dtype=np.float32),
            "traffic_light_state": traffic_light_state,
            "yaw_rate": np.array([yaw_rate], dtype=np.float32),
            "roll": np.array([roll], dtype=np.float32),
            "pitch": np.array([pitch], dtype=np.float32),
            'achieved_goal' : achieved_goal,
            'desired_goal' : desired_goal
        }
        return observation


    def setup_agent(self, vehicle, route):
        # BasicAgent or similar can be used, or directly manipulate with Traffic Manager
        agent = BasicAgent(vehicle)
        agent.set_global_plan(route)  # Convert locations to waypoints if necessary
        vehicle.set_autopilot(True)
        # Traffic manager settings if required
        self.traffic_manager.ignore_lights_percentage(vehicle, 0)  # Fully obey traffic lights

    def _get_start_transform(self, offset=0):
        """
        Get a starting transform at a random spawn point.
        Optionally, add an offset to spawn another vehicle ahead or behind the selected spawn point.
        """
        base_transform = random.choice(self.map.get_spawn_points())
        if offset != 0:
            # Calculate the forward direction from the transform rotation
            forward_vector = carla.Location(x=math.cos(math.radians(base_transform.rotation.yaw)),
                                            y=math.sin(math.radians(base_transform.rotation.yaw)))
            # Scale the forward vector by the offset and add to the base location
            offset_location = carla.Location(forward_vector.x * offset, forward_vector.y * offset, 0)
            # Create a new transform with the offset
            new_location = carla.Location(base_transform.location.x + offset_location.x,
                                        base_transform.location.y + offset_location.y,
                                        base_transform.location.z)
            return carla.Transform(new_location, base_transform.rotation)
        return base_transform   

    def initialize_traffic_manager(self):
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)
        
    def spawn_and_assign_routes(self, vehicle_spawn_indices, route_indices):
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        route = self.define_route_by_indices(route_indices)
        vehicles = []

        for index in vehicle_spawn_indices:
            blueprint = random.choice(blueprints)
            spawn_point = spawn_points[index]
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -100)  # Slower by 30%
                self.traffic_manager.set_path(vehicle, route)
                vehicles.append(vehicle)
        return vehicles
    def define_route_by_indices(self, indices):
        spawn_points = self.world.get_map().get_spawn_points()
        route = [spawn_points[index].location for index in indices]
        return route

    def cleanup(self):
        for vehicle in self.spawned_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.spawned_vehicles = []



    print("finished")      
    
        