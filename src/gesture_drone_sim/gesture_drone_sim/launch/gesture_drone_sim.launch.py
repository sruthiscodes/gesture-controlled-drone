from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('gesture_drone_sim')
    
    # Path to the URDF file
    urdf_file = os.path.join(pkg_share, 'urdf', 'improved_drone.urdf')
    
    # Start Gazebo
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )
    
    # Spawn the drone
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'improved_drone', '-file', urdf_file],
        output='screen'
    )
    
    # Start the gesture controller
    gesture_controller = Node(
        package='gesture_drone_sim',
        executable='gesture_drone_controller',
        name='gesture_controller',
        output='screen'
    )
    
    return LaunchDescription([
        gazebo,
        spawn_entity,
        gesture_controller
    ]) 