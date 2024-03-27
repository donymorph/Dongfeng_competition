#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""
from __future__ import print_function
import numpy as np
import carla
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
import pygame
import math
import random


def get_entry_point():
         return 'NpcAgent'
class HumanInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height, side_scale, left_mirror=False, right_mirror=False):
        self._width = width
        self._height = height
        self._scale = side_scale
        self._surface = None
        self.current_waypoint_index = 0

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # Process sensor data
        image_center = input_data['Center'][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))

        
        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS
        self.world = CarlaDataProvider.get_world()
        self._agent = None
        self.agent_engaged = False
        self.camera_width = 1280
        self.camera_height = 720
        self._side_scale = 0.3
        self._left_mirror = False
        self._right_mirror = False

        self._hic = HumanInterface(
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror
        )
        self._clock = pygame.time.Clock()

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 2.00, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self.camera_width, 'height': self.camera_height, 'fov': 110, 'id': 'Center'},
        ]

        return sensors
    def draw_next_waypoints(self):
        if self.plan and self.current_waypoint_index < len(self.plan):
            # Determine the last waypoint index to draw (up to 10 waypoints ahead)
            last_waypoint_index = min(self.current_waypoint_index + 5, len(self.plan))
            
            for i in range(self.current_waypoint_index, last_waypoint_index - 1):  # Adjust loop to ensure we have a next waypoint
                current_waypoint, _ = self.plan[i]
                next_waypoint, _ = self.plan[i + 1]  # Get the next waypoint to draw a line to
                
                # Locations of the current and next waypoints
                current_location = current_waypoint.transform.location + carla.Location(z=0.1)
                next_location = next_waypoint.transform.location + carla.Location(z=0.1)
                
                # Optionally draw the current waypoint as a point
                self.world.debug.draw_point(current_location, size=0.1, color=carla.Color(0, 255, 0), life_time=1.0)
                #self.world.debug.draw_string(current_location, '.', draw_shadow=False, color=carla.Color(255, 0, 0), life_time=1.0, persistent_lines=True)
                
                # Draw a line to the next waypoint
                self.world.debug.draw_line(current_location, next_location, thickness=0.1, color=carla.Color(0, 255, 0), life_time=1.0)

            # Draw the last waypoint in the sequence
            if last_waypoint_index > self.current_waypoint_index:
                last_waypoint_location = self.plan[last_waypoint_index - 1][0].transform.location + carla.Location(z=0.1)
                self.world.debug.draw_point(last_waypoint_location, size=0.1, color=carla.Color(0, 255, 0), life_time=1.0)
                #self.world.debug.draw_string(last_waypoint_location, '.', draw_shadow=False, color=carla.Color(255, 0, 0), life_time=1.0, persistent_lines=True)

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation. 
        """
        self._clock.tick_busy_loop(20)
        self.agent_engaged = True
        self._hic.run_interface(input_data)
        if not self._agent:

            # Search for the ego actor
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break

            if not hero_actor:
                return carla.VehicleControl()

            # Add an agent that follows the route to the ego
            self._agent = BasicAgent(hero_actor, 50)

            plan = []
            self.plan = plan
            self.current_waypoint_index = 0
            prev_wp = None
            for transform, _ in self._global_plan_world_coord:
                wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                if prev_wp:
                    plan.extend(self._agent.trace_route(prev_wp, wp))
                prev_wp = wp
            self._agent.set_global_plan(plan)
            print(plan[1])
            
        if self.plan and self.current_waypoint_index < len(self.plan):
            vehicle_location = self._agent._vehicle.get_location()  # Corrected way to get vehicle location
            current_waypoint, _ = self.plan[self.current_waypoint_index]
            waypoint_location = current_waypoint.transform.location

            if vehicle_location.distance(waypoint_location) < 2.0:  # Check if close to current waypoint
                self.current_waypoint_index += 1  # Move to the next waypoint

        self.draw_next_waypoints()  # Always attempt to draw the next waypoint
                


        return self._agent.run_step() if self._agent else carla.VehicleControl()
