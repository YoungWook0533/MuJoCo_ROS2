from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory and paths
    robot_share_dir = get_package_share_directory('mujoco_ros2')
    robot_urdf = os.path.join(robot_share_dir, 'models', 'fr3_xls.urdf')
    rviz_config = os.path.join(robot_share_dir, 'config', 'display.rviz')

    # Read URDF file
    with open(robot_urdf, 'r') as urdf_file:
        robot_description = urdf_file.read()

    # Declare launch argument to control fr3_position_control node
    launch_arg = DeclareLaunchArgument(
        'launch_fr3_position_control',
        default_value='true',
        description='Set to true to launch fr3_position_control.py'
    )

    # Nodes
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

    arm_controller = Node(
        package='mujoco_ros2',
        executable='fr3_position_control',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_fr3_position_control'))
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    # Return launch description with all nodes
    return LaunchDescription([
        launch_arg,
        robot_state_publisher,
        mujoco_sim,
        base_controller,
        arm_controller,
        rviz
    ])
