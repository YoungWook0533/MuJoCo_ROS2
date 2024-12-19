from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    robot_share_dir = get_package_share_directory('mujoco_ros2')
    robot_urdf = os.path.join(robot_share_dir, 'models', 'fr3_xls.urdf')

    rviz_config = os.path.join(robot_share_dir, 'rviz', 'display.rviz')
    
    with open(robot_urdf, 'r') as urdf_file:
        robot_description = urdf_file.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    mujoco_sim = Node(
        package='mujoco_ros2',
        executable='mujoco_sim',
        output='screen',
    )

    base_controller = Node(
        package='mujoco_ros2',
        executable='base_controller',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        mujoco_sim,
        base_controller,
        rviz
    ])
