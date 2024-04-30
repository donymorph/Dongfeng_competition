import numpy as np

# ==============================================================================
# -- import route planning (copied and modified from CARLA 0.9.4's PythonAPI) --
# ==============================================================================
import carla
from navigation.local_planner import RoadOption
from navigation.global_route_planner import GlobalRoutePlanner
from tools.misc import vector
from utilities.utils import load_route_from_xmlnew, get_all_route_ids, load_route_from_xmlold
import random
from collections import deque
# def compute_route_waypoints(world_map, start_waypoint, end_waypoint, resolution=1.0, plan=None):
#     """
#         Returns a list of (waypoint, RoadOption)-tuples that describes a route
#         starting at start_waypoint, ending at end_waypoint.

#         start_waypoint (carla.Waypoint):
#             Starting waypoint of the route
#         end_waypoint (carla.Waypoint):
#             Destination waypoint of the route
#         resolution (float):
#             Resolution, or lenght, of the steps between waypoints
#             (in meters)
#         plan (list(RoadOption) or None):
#             If plan is not None, generate a route that takes every option as provided
#             in the list for every intersections, in the given order.
#             (E.g. set plan=[RoadOption.STRAIGHT, RoadOption.LEFT, RoadOption.RIGHT]
#             to make the route go straight, then left, then right.)
#             If plan is None, we use the GlobalRoutePlanner to find a path between
#             start_waypoint and end_waypoint.
#     """

#     if plan is None:
#         # Setting up global router
#         grp = GlobalRoutePlanner(world_map, resolution)
        
#         # Obtain route plan
#         route = grp.trace_route(
#             start_waypoint.transform.location,
#             end_waypoint.transform.location)
#     else:
#         # Compute route waypoints
#         route = []
#         current_waypoint = start_waypoint
#         for i, action in enumerate(plan):
#             # Generate waypoints to next junction
#             wp_choice = [current_waypoint]
#             while len(wp_choice) == 1:
#                 current_waypoint = wp_choice[0]
#                 route.append((current_waypoint, RoadOption.LANEFOLLOW))
#                 wp_choice = current_waypoint.next(resolution)

#                 # Stop at destination
#                 if i > 0 and current_waypoint.transform.location.distance(end_waypoint.transform.location) < resolution:
#                     break

#             if action == RoadOption.VOID:
#                 break

#             # Make sure that next intersection waypoints are far enough
#             # from each other so we choose the correct path
#             step = resolution
#             while len(wp_choice) > 1:
#                 wp_choice = current_waypoint.next(step)
#                 wp0, wp1 = wp_choice[:2]
#                 if wp0.transform.location.distance(wp1.transform.location) < resolution:
#                     step += resolution
#                 else:
#                     break

#             # Select appropriate path at the junction
#             if len(wp_choice) > 1:
#                 # Current heading vector
#                 current_transform = current_waypoint.transform
#                 current_location = current_transform.location
#                 projected_location = current_location + \
#                     carla.Location(
#                         x=np.cos(np.radians(current_transform.rotation.yaw)),
#                         y=np.sin(np.radians(current_transform.rotation.yaw)))
#                 v_current = vector(current_location, projected_location)

#                 direction = 0
#                 if action == RoadOption.LEFT:
#                     direction = 1
#                 elif action == RoadOption.RIGHT:
#                     direction = -1
#                 elif action == RoadOption.STRAIGHT:
#                     direction = 0
#                 select_criteria = float("inf")

#                 # Choose correct path
#                 for wp_select in wp_choice:
#                     v_select = vector(
#                         current_location, wp_select.transform.location)
#                     cross = float("inf")
#                     if direction == 0:
#                         cross = abs(np.cross(v_current, v_select)[-1])
#                     else:
#                         cross = direction * np.cross(v_current, v_select)[-1]
#                     if cross < select_criteria:
#                         select_criteria = cross
#                         current_waypoint = wp_select

#                 # Generate all waypoints within the junction
#                 # along selected path
#                 route.append((current_waypoint, action))
#                 current_waypoint = current_waypoint.next(resolution)[0]
#                 while current_waypoint.is_intersection:
#                     route.append((current_waypoint, action))
#                     current_waypoint = current_waypoint.next(resolution)[0]
#         assert route

#     return route

def generate_route(self, xml_file_path):
        # Fetch all route IDs and select one randomly
        route_ids = get_all_route_ids(xml_file_path)
        if not route_ids:
            raise ValueError("No routes found in the XML file.")
        selected_route_id = random.choice(route_ids)
        
        # Load waypoints for the selected route
        waypoints = load_route_from_xmlnew(xml_file_path, selected_route_id)

        # Initialize route tracking variables
        self.route_waypoints = []
        color = carla.Color(255, 255, 0)  # Yellow color
        life_time = 0  # Persistent lines, set to a positive value for temporary lines

        # Generate the detailed route by processing each segment between consecutive waypoints
        for i in range(len(waypoints) - 1):
            start_location = waypoints[i]
            end_location = waypoints[i + 1]
            start_waypoint = self.map.get_waypoint(start_location)
            end_waypoint = self.map.get_waypoint(end_location)
            

            # Compute route between current and next waypoint
            segment_waypoints = compute_route_waypoints(self.map, start_waypoint, end_waypoint, resolution=1.0)
            self.route_waypoints.extend(segment_waypoints)  # Extend the list with the new segment
        for i in range(len(self.route_waypoints) - 1):
                start_wp, _ = self.route_waypoints[i]
                end_wp, _ = self.route_waypoints[i + 1]    
        # Draw a line between each pair of waypoints
                #self.world.debug.draw_line(start_wp.transform.location, end_wp.transform.location, thickness=0.1, color=color, life_time=life_time, persistent_lines=True)
                #self.world.debug.draw_point(start_wp.transform.location + carla.Location(z=0.5), size=0.1, color=color, life_time=life_time, persistent_lines=True)
            #print(self.route_waypoints)
        # Setup the vehicle at the starting point of the first segment
        if self.route_waypoints:
            start_wp, _ = self.route_waypoints[0]  # Assuming the first tuple contains the start waypoint
            self.vehicle.set_transform(start_wp.transform)

        # Re-enable physics for the vehicle after setting the route
        self.vehicle.set_simulate_physics(True)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.distance_from_center_history = deque(maxlen=30)

def compute_route_waypoints(world_map, start_waypoint, end_waypoint, resolution=1.0, plan=None):
    if plan is None:
        grp = GlobalRoutePlanner(world_map, resolution)
        route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
    else:
        route = []
        current_waypoint = start_waypoint
        for action in plan:
            if action == RoadOption.VOID:
                break

            if action in (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT):
                route.extend(generate_straight_route(current_waypoint, action, resolution))

            elif action in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                current_waypoint = change_lane(current_waypoint, action == RoadOption.CHANGELANELEFT)
                if current_waypoint is None:
                    raise ValueError("Lane change requested but not possible.")
                route.append((current_waypoint, action))

            # Follow the lane
            current_waypoint = follow_lane(current_waypoint, resolution)

    return route

def generate_straight_route(waypoint, action, resolution):
    route = []
    while not waypoint.is_junction:
        route.append((waypoint, RoadOption.LANEFOLLOW))
        next_waypoints = waypoint.next(resolution)
        waypoint = select_next_waypoint(next_waypoints, action)
    return route

def select_next_waypoint(next_waypoints, action):
    if action == RoadOption.STRAIGHT:
        return next_waypoints[0]  # Simplified selection for demo purposes
    elif action == RoadOption.LEFT:
        return min(next_waypoints, key=lambda wp: -wp.transform.rotation.yaw)
    elif action == RoadOption.RIGHT:
        return max(next_waypoints, key=lambda wp: wp.transform.rotation.yaw)

def follow_lane(waypoint, resolution):
    next_waypoints = waypoint.next(resolution)
    return next_waypoints[0] if next_waypoints else waypoint

def change_lane(waypoint, to_left):
    if to_left:
        return waypoint.get_left_lane() if waypoint.lane_change & carla.LaneChange.Left else None
    else:
        return waypoint.get_right_lane() if waypoint.lane_change & carla.LaneChange.Right else None
