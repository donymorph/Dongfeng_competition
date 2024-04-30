import carla
import random
from collections import deque
from xml.etree import ElementTree as ET
import time
from navigation.local_planner import RoadOption
from navigation.global_route_planner import GlobalRoutePlanner



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

# Replace references to carla.RoadOption with just RoadOption in your script

def get_all_route_ids(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return [route.get('id') for route in root.findall('.//route')]

def load_route_from_xmlnew(xml_file_path, route_id):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    route = root.find(f".//route[@id='{route_id}']")
    if route is None:
        raise ValueError(f"Route with id {route_id} not found")
    waypoints = []
    for waypoint in route.findall('waypoint'):
        x = float(waypoint.get('x'))
        y = float(waypoint.get('y'))
        z = float(waypoint.get('z'))
        waypoints.append(carla.Location(x=x, y=y, z=z))
    return waypoints

def generate_route(world, vehicle, xml_file_path):
    route_ids = get_all_route_ids(xml_file_path)
    if not route_ids:
        raise ValueError("No routes found in the XML file.")
    selected_route_id = random.choice(route_ids)
    waypoints = load_route_from_xmlnew(xml_file_path, selected_route_id)
    map = world.get_map()
    route_waypoints = []
    for i in range(len(waypoints) - 1):
        start_location = waypoints[i]
        end_location = waypoints[i + 1]
        start_waypoint = map.get_waypoint(start_location)
        end_waypoint = map.get_waypoint(end_location)

        segment_waypoints = compute_route_waypoints(map, start_waypoint, end_waypoint, resolution=1.0)
        route_waypoints.extend(segment_waypoints)  # Extend the list with the new segment
        # Here, compute_route_waypoints needs to be correctly defined or replaced with appropriate logic.
        #route_waypoints.append((start_waypoint, end_waypoint))  # Simplified for demonstration
    return route_waypoints

def draw_route(world, route_waypoints):
    """
    Draws the route in the simulation, assuming each entry in route_waypoints is a tuple containing a carla.Waypoint and a RoadOption.

    Parameters:
        world (carla.World): The CARLA world context.
        route_waypoints (list): A list of tuples, each containing a carla.Waypoint and a RoadOption.
    """
    previous_location = None  # Keep track of the last location to draw lines between waypoints
    for waypoint, _ in route_waypoints:
        # Get waypoint location, raised slightly above the road surface
        current_location = waypoint.transform.location + carla.Location(z=0.5)

        # Draw the waypoint as a point
        world.debug.draw_point(current_location, size=0.1, color=carla.Color(255, 0, 0), life_time=200.0, persistent_lines=True)

        # If there's a previous waypoint, draw a line from it to the current waypoint
        if previous_location:
            world.debug.draw_line(previous_location, current_location, thickness=0.1, color=carla.Color(255, 165, 0), life_time=200.0)
        
        # Update previous_location to the current location
        previous_location = current_location
def try_spawn_vehicle(world, blueprint, spawn_transform, retries=10, delay=0.5):
    """ Tries to spawn a vehicle, adjusting the spawn point slightly on each failure. """
    for attempt in range(retries):
        vehicle = world.try_spawn_actor(blueprint, spawn_transform)
        if vehicle is not None:
            return vehicle
        else:
            print(f"Spawn attempt {attempt + 1} failed, retrying...")
            spawn_transform.location.z += 0.1  # Slightly raise the spawn position to avoid collision
            time.sleep(delay)
    raise RuntimeError("Failed to spawn vehicle after several attempts.")

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Obtain the blueprint library and select the Tesla model 3
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    
    # Load the route from the XML file and calculate the waypoints
    xml_file_path = 'RL_SB3_new/routes/dongfeng.xml'
    route_waypoints = generate_route(world, None, xml_file_path)  # No vehicle needed yet for generating route
    
    if not route_waypoints:
        print("No route waypoints were generated.")
        return

    # Use the first waypoint's location as the spawn point
    start_location = route_waypoints[0][0].transform.location
    spawn_point = carla.Transform(start_location, carla.Rotation(yaw=route_waypoints[0][0].transform.rotation.yaw))
    print(spawn_point)
    # Spawn the vehicle at the first waypoint of the route
    try:
        vehicle = try_spawn_vehicle(world, vehicle_bp, spawn_point)
    except RuntimeError as e:
        print(e)
        return

    # Set the vehicle to follow the waypoints manually or via autopilot if capable
    vehicle.set_autopilot(True)  # This depends on CARLA's autopilot ability to follow specific routes

    # Draw the route for visual reference
    draw_route(world, route_waypoints)

    try:
        # Run the simulation long enough to observe the vehicle behavior
        import time
        time.sleep(120)  # Run simulation for 120 seconds or as needed
    finally:
        # Clean up and destroy the vehicle to free resources
        vehicle.destroy()

if __name__ == '__main__':
    main()