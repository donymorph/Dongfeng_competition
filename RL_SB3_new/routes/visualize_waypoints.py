import carla
import xml.etree.ElementTree as ET

# Connect to the CARLA Simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # seconds
world = client.get_world()

# Function to parse the XML and return waypoints
def get_waypoints_from_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    waypoints = []
    for waypoint in root.findall('.//waypoint'):
        x = float(waypoint.get('x'))
        y = float(waypoint.get('y'))
        z = float(waypoint.get('z'))  # Assuming there's a Z coordinate
        waypoints.append(carla.Location(x=x, y=y, z=z))
    return waypoints

# Function to draw the waypoints in CARLA
def draw_waypoints(world, waypoints):
    # Draw each waypoint
    for waypoint in waypoints:
        world.debug.draw_string(waypoint, 'X', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                persistent_lines=True)

    # Draw lines between waypoints
    if len(waypoints) > 1:
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            world.debug.draw_line(start, end, thickness=0.5, color=carla.Color(r=0, g=255, b=0), life_time=10.0, persistent_lines=True)

# Main execution
xml_file_path = 'RL_SB3_new/routes/dongfeng.xml'  # Update this path
waypoints = get_waypoints_from_xml(xml_file_path)
draw_waypoints(world, waypoints)
