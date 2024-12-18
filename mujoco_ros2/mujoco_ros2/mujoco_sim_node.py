import time
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

joint = [
    "base_x_slide_joint",
    "base_y_slide_joint",
    "base_z_hinge_joint",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.paused = False

        # Command arrays
        self.joint_commands = [0.0] * len(joint)

        # Publisher for joint states
        self.pub_jointstate = self.create_publisher(JointState, "/joint_states", 10)

        # Subscriber for joint commands
        self.create_subscription(Float64MultiArray, '/joint_commands', self.command_callback, 10)

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path('/home/kimyeonguk/mjpy_ros2_ws/src/MuJoCo_ROS2/mujoco_ros2/models/mobile_fr3_original.xml')
        self.data = mujoco.MjData(self.model)
        self.get_logger().info("MuJoCo simulation node initialized")

        # Map actuators for joints
        self.joint_to_actuator_map = self._create_joint_to_actuator_map()

    def _create_joint_to_actuator_map(self):
        """Create a mapping between joint names and actuator indices."""
        joint_to_actuator = {}
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id][0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and joint_name in joint:
                joint_to_actuator[joint_name] = actuator_id
        return joint_to_actuator

    def command_callback(self, msg):
        """Callback for the joint_commands topic."""
        if len(msg.data) != len(joint):  # Ensure array size matches the expected number of joints
            self.get_logger().warn(f"Invalid command size: {len(msg.data)}. Expected {len(joint) + 1}.")
            return

        # Update joint commands
        self.joint_commands = msg.data[:len(joint)]

    # Pause simulation
    def key_callback(self, keycode):
        """Keyboard callback for interacting with the simulation."""
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def mujoco(self):
        """Main simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                          show_left_ui=False, 
                                          show_right_ui=True, 
                                          key_callback=self.key_callback) as viewer:
            while True:
                step_start = time.time()

                # For processing messages in subscribed topics
                rclpy.spin_once(self, timeout_sec=0.0)

                # Update joint controls
                self.update_joints()

                # Publish joint states
                self.joint_state_publisher(self.pub_jointstate)

                # Step simulation if not paused
                if not self.paused:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                # Synchronize simulation time
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def update_joints(self):
        """Set control signals for all joints."""
        for i, command in enumerate(self.joint_commands):
            joint_name = joint[i]
            actuator_id = self.joint_to_actuator_map.get(joint_name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = command
            else:
                self.get_logger().warn(f"Actuator not found for joint {joint_name}.")

    def joint_state_publisher(self, pub_jointstate):
        """Publish joint states."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = joint
        joint_msg.position = [self.data.qpos[self.model.jnt_qposadr[joint.index(j)]] for j in joint]
        joint_msg.velocity = [self.data.qvel[self.model.jnt_dofadr[joint.index(j)]] for j in joint]
        joint_msg.effort = [self.data.qfrc_smooth[self.model.jnt_dofadr[joint.index(j)]] for j in joint]

        pub_jointstate.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()

    try:
        node.mujoco()
    except KeyboardInterrupt:
        node.get_logger().info("Simulation interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# ros2 topic pub /joint_commands std_msgs/msg/Float64MultiArray "data: [0.0, 0.0, 0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]"
