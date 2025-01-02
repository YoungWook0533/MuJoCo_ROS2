import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
import numpy as np


class EndEffectorPositionControlNode(Node):
    def __init__(self):
        super().__init__('end_effector_position_control_node')

        # Load the URDF model into Pinocchio
        urdf_path = "/home/yeonguk/mjpy_ros2_ws/src/MuJoCo_ROS2/mujoco_ros2/models/fr3_xls_pinocchio.urdf"
        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()

        # End-effector frame
        self.ee_frame_name = "panda_leftfinger"  # Update to the correct frame name
        self.ee_frame_id = self.robot.getFrameId(self.ee_frame_name)
        if self.ee_frame_id == -1:
            self.get_logger().error(f"End-effector frame '{self.ee_frame_name}' not found in the robot model!")
            return

        # Joint names
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
        ]
        self.num_joints = len(self.joint_names)

        # Initialize joint state arrays
        self.q = np.zeros(self.num_joints)        # Current joint positions
        self.q_dot = np.zeros(self.num_joints)    # Current joint velocities
        self.torques = np.zeros(self.num_joints)  # Computed joint torques

        # Control gains
        self.Kp = np.eye(6) * 100.0  # Proportional gains for task space
        self.Kv = np.eye(6) * 20.0   # Derivative gains

        # Desired pose
        self.desired_pose = None  # Desired pose will be updated via the /ee_position_command topic

        # Subscribers
        self.create_subscription(JointState, "/joint_states", self.joint_states_callback, 10)
        self.create_subscription(PoseStamped, "/ee_position_command", self.ee_position_command_callback, 10)

        # Publisher to /manipulator_commands
        self.torque_pub = self.create_publisher(Float64MultiArray, "/manipulator_commands", 10)

        # Timer to compute and publish torques at 100 Hz
        self.timer = self.create_timer(0.01, self.compute_and_publish_torques)

        self.get_logger().info("End Effector Position Control Node initialized.")

    def joint_states_callback(self, msg):
        """Update current joint positions and velocities."""
        try:
            for i, name in enumerate(self.joint_names):
                index = msg.name.index(name)
                self.q[i] = msg.position[index]
                self.q_dot[i] = msg.velocity[index]
        except ValueError as e:
            self.get_logger().warn(f"Joint {name} not found in /joint_states: {e}")

    def ee_position_command_callback(self, msg):
        """Callback to update the desired end-effector pose."""
        self.desired_pose = {
            "translation": np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            "rotation": pin.Quaternion(msg.pose.orientation.w, msg.pose.orientation.x,
                                        msg.pose.orientation.y, msg.pose.orientation.z).toRotationMatrix()
        }

    def compute_orientation_error(self, R_current, R_desired):
        """Compute orientation error as a 3D vector."""
        R_error = np.dot(R_desired.T, R_current)  # Error rotation matrix
        skew_symmetric = (R_error - R_error.T) / 2.0
        orientation_error = np.array([
            -skew_symmetric[1, 2],
            skew_symmetric[0, 2],
            -skew_symmetric[0, 1]
        ])
        return orientation_error

    def compute_and_publish_torques(self):
        """Compute inverse dynamics torques for end-effector position control."""
        if self.desired_pose is None:
            # No desired pose has been set yet
            return

        # Forward kinematics to compute the current end-effector pose
        pin.forwardKinematics(self.robot, self.data, self.q, self.q_dot)
        pin.updateFramePlacement(self.robot, self.data, self.ee_frame_id)

        current_pose = self.data.oMf[self.ee_frame_id]
        current_translation = current_pose.translation
        current_rotation = current_pose.rotation

        # Log the current end-effector pose
        self.get_logger().info(f"Current EE Translation: {current_translation}")
        self.get_logger().info(f"Current EE Rotation (as matrix): \n{current_rotation}")

        # Compute errors in translation and orientation
        translation_error = self.desired_pose["translation"] - current_translation
        orientation_error = self.compute_orientation_error(current_rotation, self.desired_pose["rotation"])

        # Stack errors into a 6D task-space error
        task_error = np.hstack((translation_error, orientation_error))

        # Compute task-space velocity error
        J = pin.computeFrameJacobian(self.robot, self.data, self.q, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        ee_velocity = J @ self.q_dot
        task_velocity_error = -ee_velocity

        # PD control law in task space
        task_acceleration = self.Kp @ task_error + self.Kv @ task_velocity_error

        # Map task accelerations to joint space
        q_ddot_desired = np.linalg.pinv(J) @ task_acceleration

        # Compute dynamics matrices
        M = pin.crba(self.robot, self.data, self.q)
        C = pin.computeCoriolisMatrix(self.robot, self.data, self.q, self.q_dot)
        g = pin.computeGeneralizedGravity(self.robot, self.data, self.q)

        # Compute torques using the equation: τ = M * q_ddot + C * q_dot + g
        self.torques = M @ q_ddot_desired + C @ self.q_dot + g

        # Publish torques
        torque_msg = Float64MultiArray()
        torque_msg.data = self.torques.tolist()
        self.torque_pub.publish(torque_msg)

        # Debug output
        self.get_logger().info(f"Translation error: {translation_error}, Orientation error: {orientation_error}, Torques: {self.torques}")



def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorPositionControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
