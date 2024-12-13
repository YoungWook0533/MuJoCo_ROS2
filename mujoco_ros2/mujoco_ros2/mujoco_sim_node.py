import time
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')
        self.paused = False

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path('/home/kimyeonguk/models/mobile_fr3/fr3_xl_steel_real/mobile_fr3_original.xml')
        self.data = mujoco.MjData(self.model)
        self.get_logger().info("MuJoCo simulation node initialized")

    def key_callback(self, keycode):
        """Keyboard callback for interacting with the simulation."""
        if chr(keycode) == ' ':
            self.paused = not self.paused

    def run_simulation(self):
        """Main simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                          show_left_ui=True, 
                                          show_right_ui=True, 
                                          key_callback=self.key_callback) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < 60:
                step_start = time.time()

                if not self.paused:
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                # Example: Toggle contact points dynamically
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time)

                # Synchronize simulation time
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()

    try:
        node.run_simulation()
    except KeyboardInterrupt:
        node.get_logger().info("Simulation interrupted by user.")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
