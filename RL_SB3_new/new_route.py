import carla
import random
import xml.etree.ElementTree as ET
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.local_planner import LocalPlanner
from navigation.basic_agent import BasicAgent
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
def setup_vehicle(world, spawn_point):
    """Spawn a vehicle in the world at the given spawn_point."""
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]  # Example: Use a Tesla Model 3
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle
def generate_route(world, xml_file_path, route_id, sampling_resolution=1.0):
    waypoints = load_route_from_xmlnew(xml_file_path, route_id)
    global_route_planner = GlobalRoutePlanner(world.get_map(), sampling_resolution)
    route_trace = []  # This will store the complete route
    for i in range(len(waypoints) - 1):
        segment_trace = global_route_planner.trace_route(waypoints[i], waypoints[i + 1])
        route_trace.extend(segment_trace)  # Append each segment to the main route
    return route_trace

def draw_route(world, route_waypoints):
    previous_location = None
    for waypoint, _ in route_waypoints:
        current_location = waypoint.transform.location + carla.Location(z=0.5)
        world.debug.draw_point(current_location, size=0.1, color=carla.Color(255, 0, 0), life_time=200.0, persistent_lines=True)
        if previous_location:
            # world.debug.draw_line(previous_location, current_location, thickness=0.1, color=carla.Color(255, 165, 0), life_time=200.0)
            previous_location = current_location


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    xml_file_path = 'RL_SB3_new/routes/dongfeng.xml'
    route_ids = get_all_route_ids(xml_file_path)
    selected_route_id = random.choice(route_ids)

    # Spawn a vehicle
    spawn_point = carla.Transform(carla.Location(x=-41.49, y=111.94, z=1), carla.Rotation(pitch=0, yaw=-90, roll=0))  # Example spawn point
    vehicle = setup_vehicle(world, spawn_point)

    # Generate route
    route_waypoints = generate_route(world, xml_file_path, selected_route_id)
    draw_route(world, route_waypoints)
    # Setup Local Planner
    local_planner = BasicAgent(vehicle)
    local_planner.set_global_plan(route_waypoints, clean_queue=True)
    LocalPlanner.run_step
    # Simulation loop
    try:
        while not local_planner.done():
            control = local_planner.run_step(debug=True)
            vehicle.apply_control(control)
    finally:
        print("Simulation ended.")
        vehicle.destroy()

if __name__ == "__main__":
    main()