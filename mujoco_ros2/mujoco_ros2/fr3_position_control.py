import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np


class JointPositionControlNode(Node):
    def __init__(self):
        super().__init__('inverse_dynamics_position_control_node')

        # Load the URDF model into Pinocchio
        urdf_path = "/home/yeonguk/mjpy_ros2_ws/src/MuJoCo_ROS2/mujoco_ros2/models/fr3_xls_pinocchio.urdf"
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()

        # Joint names (must match /joint_states order)
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
        ]
        self.num_joints = len(self.joint_names)

        # Initialize joint state arrays
        self.q = np.zeros(self.num_joints)        # Current joint positions
        self.q_dot = np.zeros(self.num_joints)    # Current joint velocities
        self.q_ddot = np.zeros(self.num_joints)   # Desired joint accelerations
        self.torques = np.zeros(self.num_joints)  # Computed joint torques

        # Control gains (adjust gains for specific joints)
        self.Kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # Proportional gains
        self.Kv = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])         # Derivative gains

        # Desired state initialized with initial state
        self.q_desired = np.array([
            0.0,                  # joint1
            -0.785398163397,      # joint2
            0.0,                  # joint3
            -2.35619449019,       # joint4
            0.0,                  # joint5
            1.57079632679,        # joint6
            0.785398163397        # joint7
        ])
        self.q_dot_desired = np.zeros(self.num_joints)    # Target joint velocities
        self.q_ddot_desired = np.zeros(self.num_joints)   # Target joint accelerations

        # Subscribers
        self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        self.create_subscription(Float64MultiArray, "/position_command", self.position_command_callback, 10)

        # Publisher to /manipulator_commands
        self.torque_pub = self.create_publisher(Float64MultiArray, "/manipulator_commands", 10)

        # Compute and publish torques at 1000 Hz
        self.timer = self.create_timer(0.001, self.compute_and_publish_torques)

        self.get_logger().info("Inverse Dynamics Position Control Node initialized.")

    def joint_states_callback(self, msg):
        """Update current joint positions and velocities."""
        try:
            for i, name in enumerate(self.joint_names):
                index = msg.name.index(name)
                self.q[i] = msg.position[index]
                self.q_dot[i] = msg.velocity[index]
        except ValueError as e:
            self.get_logger().warn(f"Joint {name} not found in /joint_states: {e}")

    def position_command_callback(self, msg):
        """Callback to update desired joint positions directly."""
        if len(msg.data) != self.num_joints:
            self.get_logger().warn("Received position command with invalid joint count.")
            return

        # Update the desired joint positions
        self.q_desired = np.array(msg.data)

    def compute_and_publish_torques(self):
        """Compute inverse dynamics torques for position control."""
        # Compute position and velocity errors
        q_error = self.q_desired - self.q
        q_dot_error = self.q_dot_desired - self.q_dot

        # PD control to compute desired accelerations
        self.q_ddot = self.Kp * q_error + self.Kv * q_dot_error + self.q_ddot_desired

        # Compute the mass matrix
        M = pin.crba(self.robot, self.data, self.q)

        # Compute Coriolis and centrifugal forces
        C = pin.computeCoriolisMatrix(self.robot, self.data, self.q, self.q_dot)

        # Compute gravity forces
        g = pin.computeGeneralizedGravity(self.robot, self.data, self.q)

        # Compute torques
        self.torques = M @ self.q_ddot + C @ self.q_dot + g

        # Publish torques
        torque_msg = Float64MultiArray()
        torque_msg.data = self.torques.tolist()
        self.torque_pub.publish(torque_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointPositionControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

