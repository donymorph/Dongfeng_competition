import carla

def adjust_waypoints_to_road_surface(waypoints):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Set timeout in seconds
    world = client.get_world()
    map = world.get_map()

    adjusted_waypoints = []

    for waypoint in waypoints:
        # Create a carla.Location object for the waypoint
        location = carla.Location(x=waypoint['Location.x'], y=waypoint['Location.y'], z=waypoint['Location.z'])

        # Find the nearest waypoint on the road, projecting onto the road surface
        road_waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if road_waypoint is not None:
            adjusted_location = road_waypoint.transform.location
            # Append the adjusted waypoint to the list
            adjusted_waypoints.append(adjusted_location)
        else:
            print(f"Warning: No road waypoint found for location {location}. Skipping.")

    return adjusted_waypoints

def visualize_adjusted_waypoints(adjusted_waypoints):
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Visualize each waypoint
    for location in adjusted_waypoints:
        world.debug.draw_point(location, size = 0.1, color=carla.Color(r=255, g=0, b=0), life_time=10, persistent_lines=True)
        
    for i in range(len(waypoints) - 1):
        start_location = carla.Location(x=waypoints[i]['Location.x'], y=waypoints[i]['Location.y'], z=waypoints[i]['Location.z'])
        end_location = carla.Location(x=waypoints[i+1]['Location.x'], y=waypoints[i+1]['Location.y'], z=waypoints[i+1]['Location.z'])
        
        world.debug.draw_line(start_location, end_location, thickness=0.5, color=carla.Color(r=255, g=0, b=0), life_time=10.0, persistent_lines=True)
# Define your original waypoints here
waypoints = [
    {"Location.x": -41.6, "Location.y": 111.2, "Location.z": 1.0},
    {"Location.x": -41.3, "Location.y": 81.5, "Location.z": 1.0},
    {"Location.x": -26.2, "Location.y": 70, "Location.z": 1.0},
    {"Location.x": 25.4, "Location.y": 70, "Location.z": 1.0},
    {"Location.x": 44, "Location.y": 55.1, "Location.z": 1.0},
    {"Location.x": 43.9, "Location.y": 41.7, "Location.z": 1.0},
    {"Location.x": 56.4, "Location.y": 28.5, "Location.z": 1.0},
    {"Location.x": 81, "Location.y": 27, "Location.z": 1.0},
    {"Location.x": 106, "Location.y": -3, "Location.z": 1.0},
    {"Location.x": 106, "Location.y": -28.8, "Location.z": 1.0},
    {"Location.x": 63.7, "Location.y": -65.4, "Location.z": 1.0},
    {"Location.x": -24.7, "Location.y": -64.6, "Location.z": 1.0},
    {"Location.x": -48.6, "Location.y": -42.9, "Location.z": 1.0},
    {"Location.x": -48.4, "Location.y": 112.3, "Location.z": 1.0},
    {"Location.x": -74.5, "Location.y": 135.1, "Location.z": 1.0},
    {"Location.x": -104.1, "Location.y": 78.1, "Location.z": 1.0},
    {"Location.x": -103.7, "Location.y": 45.6, "Location.z": 1.0},
    {"Location.x": -90.3, "Location.y": 24.2, "Location.z": 1.0},
    {"Location.x": -66.6, "Location.y": 24.7, "Location.z": 1.0},
    {"Location.x": -45.5, "Location.y": -0.9, "Location.z": 1.0},
    {"Location.x": -45.3, "Location.y": -40.8, "Location.z": 1.0},
    {"Location.x": -67.9, "Location.y": -65.5, "Location.z": 1.0},
    {"Location.x": -109, "Location.y": -22.1, "Location.z": 1.0},
    {"Location.x": -111.3, "Location.y": 73.4, "Location.z": 1.0},
]

# Adjust waypoints to the road surface
adjusted_waypoints = adjust_waypoints_to_road_surface(waypoints)

# Visualize the adjusted waypoints
visualize_adjusted_waypoints(adjusted_waypoints)
