import yaml
import os
import time
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# Define the YAML file path
yaml_file_path = "/home/yeonguk/mjpy_ros2_ws/src/MuJoCo_ROS2/mujoco_ros2/config/initial_positions.yaml"

def load_initial_positions(yaml_file):
    """Load initial joint positions from a YAML file."""
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Load initial joint positions
initial_positions = load_initial_positions(yaml_file_path)

base_joints = list(initial_positions["base_joints"].keys())
manipulator_joints = list(initial_positions["manipulator_joints"].keys())
finger_joints = list(initial_positions["finger_joints"].keys())

initial_joint_positions = {**initial_positions["base_joints"], 
                           **initial_positions["manipulator_joints"], 
                           **initial_positions["finger_joints"]}

gripper_tendon = "split"

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.paused = False

        self.command_received = {
            "base": False,
            "manipulator": False,
            "gripper": False
        }

        # Command arrays
        self.base_commands = [0.0] * len(base_joints)
        self.manipulator_commands = [0.0] * len(manipulator_joints)
        self.gripper_command = 0.0

        # Publisher for joint states
        self.pub_jointstate = self.create_publisher(JointState, "/joint_states", 10)

        # Subscribers for base, manipulator, and gripper commands
        self.create_subscription(Float64MultiArray, '/base_commands', self.base_command_callback, 10)
        self.create_subscription(Float64MultiArray, '/manipulator_commands', self.manipulator_command_callback, 10)
        self.create_subscription(Float64MultiArray, '/gripper_command', self.gripper_command_callback, 10)

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path('/home/yeonguk/mjpy_ros2_ws/src/MuJoCo_ROS2/mujoco_ros2/models/mobile_fr3_original.xml')
        self.data = mujoco.MjData(self.model)
        self.get_logger().info("MuJoCo simulation node initialized")

        # Map actuators for joints and tendons
        self.joint_to_actuator_map = self._create_joint_to_actuator_map()
        self.tendon_to_actuator_map = self._create_tendon_to_actuator_map()

    def _create_joint_to_actuator_map(self):
        """Create a mapping between joint names and actuator indices."""
        joint_to_actuator = {}
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id][0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and (joint_name in base_joints or joint_name in manipulator_joints or joint_name in finger_joints):
                joint_to_actuator[joint_name] = actuator_id
        return joint_to_actuator

    def _create_tendon_to_actuator_map(self):
        """Create a mapping between tendon names and actuator indices."""
        tendon_to_actuator = {}
        for actuator_id in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if actuator_name == gripper_tendon:
                tendon_to_actuator[gripper_tendon] = actuator_id
        return tendon_to_actuator

    # Remaining code...


    def base_command_callback(self, msg):
        """Callback for the /base_commands topic."""
        if len(msg.data) != len(base_joints):  # Ensure array size matches the number of base joints
            self.get_logger().warn(f"Invalid base command size: {len(msg.data)}. Expected {len(base_joints)}.")
            return
        self.base_commands = msg.data
        self.command_received["base"] = True

    def manipulator_command_callback(self, msg):
        """Callback for the /manipulator_commands topic."""
        if len(msg.data) != len(manipulator_joints):  # Ensure array size matches the number of manipulator joints
            self.get_logger().warn(f"Invalid manipulator command size: {len(msg.data)}. Expected {len(manipulator_joints)}.")
            return
        self.manipulator_commands = msg.data
        self.command_received["manipulator"] = True

    def gripper_command_callback(self, msg):
        """Callback for the /gripper_command topic."""
        if len(msg.data) != 1:  # Ensure a single value for the gripper tendon
            self.get_logger().warn(f"Invalid gripper command size: {len(msg.data)}. Expected 1.")
            return
        self.gripper_command = msg.data[0]
        self.command_received["gripper"] = True

    def mujoco(self):
        """Main simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=False,
                                          show_right_ui=True,
                                          key_callback=self.key_callback) as viewer:

            while True:
                step_start = time.time()

                # Process messages in subscribed topics
                rclpy.spin_once(self, timeout_sec=0.0)

                # Stabilize positions if no commands received
                if not self.command_received["base"]:
                    self.stabilize_initial_positions(base_joints)
                if not self.command_received["manipulator"]:
                    self.stabilize_initial_positions(manipulator_joints)
                if not self.command_received["gripper"]:
                    self.stabilize_initial_positions(finger_joints)

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

    def stabilize_initial_positions(self, joints):
        """Stabilize joints at their initial positions."""
        for joint_name in joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_index] = initial_joint_positions[joint_name]

    def update_joints(self):
        """Set control signals for all joints and tendons."""
        # Update base joints
        for i, command in enumerate(self.base_commands):
            joint_name = base_joints[i]
            actuator_id = self.joint_to_actuator_map.get(joint_name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = command

        # Update manipulator joints
        for i, command in enumerate(self.manipulator_commands):
            joint_name = manipulator_joints[i]
            actuator_id = self.joint_to_actuator_map.get(joint_name)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = command

        # Update gripper tendon
        actuator_id = self.tendon_to_actuator_map.get(gripper_tendon)
        if actuator_id is not None:
            self.data.ctrl[actuator_id] = self.gripper_command

    def joint_state_publisher(self, pub_jointstate):
        """Publish joint states."""
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = base_joints + manipulator_joints + finger_joints

        def get_joint_id(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        joint_msg.position = [
            self.data.qpos[self.model.jnt_qposadr[get_joint_id(name)]]
            for name in joint_msg.name
        ]
        joint_msg.velocity = [
            self.data.qvel[self.model.jnt_dofadr[get_joint_id(name)]]
            for name in joint_msg.name
        ]
        joint_msg.effort = [
            self.data.qfrc_smooth[self.model.jnt_dofadr[get_joint_id(name)]]
            for name in joint_msg.name
        ]

        pub_jointstate.publish(joint_msg)

    def key_callback(self, keycode):
        """Keyboard callback for interacting with the simulation."""
        if chr(keycode) == ' ':
            self.paused = not self.paused


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
