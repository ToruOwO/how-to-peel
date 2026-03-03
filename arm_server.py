# References:
# - https://github.com/jimmyyhwu/tidybot2/blob/main/arm_server.py

import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from arm_controller import JointCompliantController, PositionController
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY, RESET_CMD_DEG
from kinova import TorqueControlledArm


class Arm:
    def __init__(self):
        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(
            speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,))
        )
        self.command_queue = queue.Queue(1)
        self.controller = None

    def reset(self, move_arm=True, object_type="long", use_position_control=False):
        # Stop low-level control
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()

        # Clear faults
        self.arm.clear_faults()

        if move_arm:
            # Reset arm configuration
            self.arm.angular_action_movement(RESET_CMD_DEG[object_type])

        # Create new instance of controller
        if use_position_control:
            self.controller = PositionController(self.command_queue)
        else:
            self.controller = JointCompliantController(self.command_queue)

        # Start low-level control
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action(self, action):
        self.command_queue.put((action["arm_qpos"], 0))

    def get_state(self):
        state = {
            "arm_qpos": self.arm.q,
        }
        return state

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)  # Wait for arm to stop moving
            self.arm.stop_cyclic()
        self.arm.disconnect()


class ArmManager(MPBaseManager):
    pass


ArmManager.register("Arm", Arm)

if __name__ == "__main__":
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    print(f"Starting arm manager server at {ARM_RPC_HOST}:{ARM_RPC_PORT}..., RPC_AUTHKEY={RPC_AUTHKEY}")
    server = manager.get_server()
    print(f"Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}")
    server.serve_forever()
