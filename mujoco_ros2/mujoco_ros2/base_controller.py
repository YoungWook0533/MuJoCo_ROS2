import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import math


class HolonomicBaseController(Node):
    def __init__(self):
        super().__init__('holonomic_base_controller')

        self.global_theta = 0.0  # Global orientation (in radians)

        # Initialize local velocities
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_theta = 0.0

        # Subscribe to /cmd_vel for desired velocities
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Subscribe to /joint_states for robot's current pose
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        # Publisher for x, y, theta joint velocities (robot frame)
        self.joint_vel_pub = self.create_publisher(
            Float64MultiArray,
            '/base_commands',
            10
        )

        # Timer for updating the state
        self.timer = self.create_timer(0.01, self.update_state)  # 10ms update

    def cmd_vel_callback(self, msg):
        """Callback to update velocities from /cmd_vel."""
        # Desired velocities in the robot frame
        self.v_x = msg.linear.x
        self.v_y = msg.linear.y
        self.v_theta = msg.angular.z

    def joint_states_callback(self, msg):
        """Callback to update the global pose from /joint_states."""

        self.global_theta = msg.position[2]

    def update_state(self):
        """Publish joint velocities."""

        # Transform velocities into robot's frame
        cos_theta = math.cos(self.global_theta)
        sin_theta = math.sin(self.global_theta)

        # Calculate effective velocities in the global frame
        x_dot = cos_theta * self.v_x - sin_theta * self.v_y
        y_dot = sin_theta * self.v_x + cos_theta * self.v_y
        theta_dot = self.v_theta

        # Publish joint velocities
        joint_vel_msg = Float64MultiArray()
        joint_vel_msg.data = [x_dot, y_dot, theta_dot]  # Local velocities
        self.joint_vel_pub.publish(joint_vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HolonomicBaseController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
