from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('gesture_drone_sim')
    print(f"Package directory: {pkg_dir}")
    
    # Kill any existing Gazebo processes
    kill_gazebo = ExecuteProcess(
        cmd=['pkill', '-f', 'gazebo'],
        output='screen'
    )
    
    # Load the URDF file
    urdf_file = os.path.join(pkg_dir, 'urdf', 'gesture_drone.urdf.xacro')
    print(f"URDF file path: {urdf_file}")
    
    # Verify URDF file exists
    if not os.path.exists(urdf_file):
        print(f"ERROR: URDF file not found at {urdf_file}")
    
    # Read URDF content once
    urdf_content = open(urdf_file, 'r').read()
    
    # Launch Gazebo with more debugging
    gazebo = ExecuteProcess(
        cmd=['gazebo', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '--verbose'],
        output='screen'
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': urdf_content,
            'publish_frequency': 30.0
        }]
    )
    
    # Spawn the robot using the spawn_entity.py script
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'gesture_drone',
            '-file', urdf_file,
            '-x', '0',
            '-y', '0',
            '-z', '1',
            '-R', '0',
            '-P', '0',
            '-Y', '0'
        ],
        output='screen'
    )
    
    # Gesture Controller with camera parameters
    gesture_controller = Node(
        package='gesture_drone_sim',
        executable='gesture_drone_controller.py',
        name='gesture_drone_controller',
        output='screen',
        parameters=[{
            'camera_id': 0,
            'camera_width': 640,
            'camera_height': 480,
            'debug_mode': True
        }]
    )
    
    # Add all nodes to the launch description with proper timing
    ld = LaunchDescription([
        kill_gazebo,
        # Add a delay after killing Gazebo
        ExecuteProcess(
            cmd=['sleep', '2'],
            output='screen'
        ),
        gazebo,
        # Add a longer delay to ensure Gazebo is fully started
        ExecuteProcess(
            cmd=['sleep', '10'],
            output='screen'
        ),
        robot_state_publisher,
        # Add a delay before spawning the entity
        ExecuteProcess(
            cmd=['sleep', '5'],
            output='screen'
        ),
        spawn_entity,
        # Add a delay before starting the gesture controller
        ExecuteProcess(
            cmd=['sleep', '5'],
            output='screen'
        ),
        gesture_controller
    ])
    
    return ld 