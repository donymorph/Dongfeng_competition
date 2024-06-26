import carla
import pygame
import numpy as np
import weakref
import xml.etree.ElementTree as ET
import math

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name

def smooth_action(old_value, new_value, smooth_factor):
    return old_value * smooth_factor + new_value * (1.0 - smooth_factor)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2].astype(int)

def distance_to_line(A, B, p):
    p[2] = 0
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])

# load routes from xml file 
def load_route_from_xmlold(xml_file_path, route_id):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find the route with the specified id
    route = root.find(f".//route[@id='{route_id}']")
    if route is None:
        raise ValueError(f"Route with id {route_id} not found")
    
    waypoints = []
    for waypoint in route.find('waypoints'):
        x = float(waypoint.get('x'))
        y = float(waypoint.get('y'))
        z = float(waypoint.get('z'))
        waypoints.append(carla.Location(x=x, y=y, z=z))
    
    return waypoints


def load_route_from_xmlnew(xml_file_path, route_id):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find the route with the specified id
    route = root.find(f".//route[@id='{route_id}']")
    if route is None:
        raise ValueError(f"Route with id {route_id} not found")
    
    waypoints = []
    for waypoint in route.findall('waypoint'):
        x = float(waypoint.get('x'))
        y = float(waypoint.get('y'))
        z = float(waypoint.get('z'))
       
        # Create a Transform if you need orientation as well
        transform = carla.Location(carla.Location(x=x, y=y, z=z))
        waypoints.append(transform)
    
    return waypoints

def get_all_route_ids(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    route_ids = [route.get('id') for route in root.findall('.//route')]
    return route_ids


def world_to_screen(world_location, camera_transform, display_dimensions):
    # Convert the 3D world location to 2D screen location considering the camera transform

    # Compute the relative location (from camera perspective)
    relative_location = world_location - camera_transform.location
    relative_location = carla.Location(
        relative_location.x * math.cos(math.radians(camera_transform.rotation.yaw)) + relative_location.y * math.sin(math.radians(camera_transform.rotation.yaw)),
        -relative_location.x * math.sin(math.radians(camera_transform.rotation.yaw)) + relative_location.y * math.cos(math.radians(camera_transform.rotation.yaw)),
        relative_location.z
    )

    # Perspective projection parameters (assuming orthographic projection)
    scale = display_dimensions[1] / (2.0 * camera_transform.location.z * math.tan(math.radians(90 / 2.0)))

    # Convert to screen coordinates
    screen_x = display_dimensions[0] / 2 + scale * relative_location.x
    screen_y = display_dimensions[1] / 2 - scale * relative_location.y

    return int(screen_x), int(screen_y)


def draw_route(display, waypoints, camera_transform, display_dimensions):
    if len(waypoints) < 2:
        return  # Need at least two waypoints to draw a route

    # Waypoints color and thickness
    color = (255, 255, 0)  # Yellow
    thickness = 2

    # Loop through the waypoints
    for i in range(len(waypoints) - 1):
        waypoint_location = waypoints[i][0].transform.location
        next_waypoint_location = waypoints[i + 1][0].transform.location

        # Convert world locations to screen locations
        screen_x, screen_y = world_to_screen(waypoint_location, camera_transform, display_dimensions)
        next_screen_x, next_screen_y = world_to_screen(next_waypoint_location, camera_transform, display_dimensions)

        # Draw a line segment between the current waypoint and the next
        pygame.draw.line(display, color, (screen_x, screen_y), (next_screen_x, next_screen_y), thickness)

