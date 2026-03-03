import time

import numpy as np

from arm_server import ArmManager
from async_sam_camera import create_async_sam_system
from constants import (
    ARM_RPC_HOST,
    ARM_RPC_PORT,
    POLICY_CONTROL_PERIOD,
    RESET_CMD,
    RPC_AUTHKEY,
)
from ik_solver import IKSolver
from netft import ForceSensor


class MockArmEnv:
    def __init__(
        self,
        cameras=None,
        camera_names=None,
        sam=False,
        calibrate=False,
        object_type="long",
        sam_model_type="b+",
        sam_fps=3.0,
        use_position_control=False,
    ):
        # NOTE: check frame!
        self.ik_solver = IKSolver(calibrate=calibrate)
        self.object_type = object_type

        # Camera objects
        self.cameras = cameras
        self.camera_names = camera_names
        self.camera_data = [None] * len(camera_names)

        if cameras is not None and sam:
            self.sam_manager = create_async_sam_system(
                cameras=cameras,
                camera_names=camera_names,
                sam_fps=sam_fps,
                model_type=sam_model_type,
                device_base="cuda",
            )
            self.sam_manager.start_all()

            self.sam = True
            self.sam_initialized = False
            print(f"Created async SAM system: {sam_fps} FPS, model: {sam_model_type}")
        else:
            self.sam = False
            self.sam_manager = None
            self.sam_initialized = False

        # Force sensor
        self.force_sensor = ForceSensor()

        # Control mode
        self.use_position_control = use_position_control

    def get_camera_data(self):
        for i in range(len(self.cameras)):
            self.camera_data[i] = self.cameras[i].get(out=self.camera_data[i])
        return self.camera_data

    def add_camera_obs(self, obs):
        if self.cameras is not None:
            self.camera_data = self.get_camera_data()
            rgb_frames = {}  # Collect RGB frames for SAM
            for camera_id, camera_name in enumerate(self.camera_names):
                obs[f"{camera_name}_rgb"] = self.camera_data[camera_id]["color"][
                    :, :, ::-1
                ].copy()
                obs[f"{camera_name}_depth"] = self.camera_data[camera_id][
                    "depth"
                ].copy()  # raw depth
                rgb_frames[camera_name] = obs[
                    f"{camera_name}_rgb"
                ]  # Store RGB for SAM input

            if self.sam:
                # Submit current RGB frames to async SAM (non-blocking)
                self.sam_manager.submit_frames(rgb_frames)

                # Get latest available masks (non-blocking)
                latest_masks = self.sam_manager.get_latest_masks()

                # Add masks to observation
                for camera_name in self.camera_names:
                    obs[f"{camera_name}_mask"] = latest_masks[camera_name]
        return obs

    def get_obs(self):
        arm_pos, arm_quat = self.ik_solver.run_fk(RESET_CMD[self.object_type])
        obs = {
            "arm_qpos": RESET_CMD[self.object_type],
            "arm_pos": arm_pos,
            "arm_quat": arm_quat,
            "force": np.zeros(6),
        }
        for camera_name in self.camera_names:
            obs[f"{camera_name}_rgb"] = np.zeros((480, 640, 3), dtype=np.uint8)
            obs[f"{camera_name}_depth"] = np.zeros((480, 640), dtype=np.float32)
            obs[f"{camera_name}_mask"] = {
                "binary_mask": np.zeros((480, 640), dtype=np.float32),
                "masks": {
                    "knife": np.zeros((480, 640), dtype=np.float32),
                    "object": np.zeros((480, 640), dtype=np.float32),
                },
                "age_ms": float("inf"),
            }

        obs = self.add_camera_obs(obs)

        return obs

    def reset(self, reset_arm=True):
        print("Resetting env...")

        if self.sam:
            self.sam_initialized = False

            # Initialize segmentation for all cameras
            self.camera_data = self.get_camera_data()
            rgb_frames = {
                camera_name: self.camera_data[camera_id]["color"][..., ::-1]
                for camera_id, camera_name in enumerate(self.camera_names)
            }

            # Initialize segmentation (blocking operation to ensure first masks are ready)
            self.sam_manager.init_all_seg(rgb_frames)
            print("Initialized async SAM segmentation for all cameras")
            self.sam_initialized = True

    def step(self, action):
        print(f"Arm step: {action}")

    def close(self):
        print("Closing env...")
        if self.sam:
            print("Stopping async SAM processing...")
            try:
                self.sam_manager.stop_all()
            except Exception as e:
                print(f"Warning: error during SAM shutdown: {e}")
        self.force_sensor.close()


class ArmEnv(MockArmEnv):
    def __init__(
        self,
        cameras=None,
        camera_names=None,
        sam=False,
        calibrate=False,
        object_type="long",
        sam_model_type="b+",
        sam_fps=3.0,
        use_position_control=False,
    ):
        super().__init__(
            cameras,
            camera_names,
            sam,
            calibrate,
            object_type,
            sam_model_type,
            sam_fps,
            use_position_control,
        )
        # RPC server connection for arm
        arm_manager = ArmManager(
            address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY
        )
        print(f"Connecting to arm RPC server at {ARM_RPC_HOST}:{ARM_RPC_PORT}...")
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception(
                "Could not connect to arm RPC server, is arm_server.py running?"
            ) from e

        # RPC proxy objects
        self.arm = arm_manager.Arm()

    def get_obs(self):
        obs = {}
        obs.update(self.arm.get_state())
        obs["arm_pos"], obs["arm_quat"] = self.ik_solver.run_fk(obs["arm_qpos"])

        obs = self.add_camera_obs(obs)

        obs["force"] = self.force_sensor.get_data()
        return obs

    def reset(self, reset_arm=True):
        print("Resetting env...")

        self.arm.reset(
            move_arm=reset_arm,
            object_type=self.object_type,
            use_position_control=self.use_position_control,
        )
        self.force_sensor.reset()

        if self.sam:
            self.sam_initialized = False

            if reset_arm:
                time.sleep(3)

                # wait until arm is near RESET_CMD before init_seg
                while True:
                    reset_done = np.allclose(
                        self.arm.get_state()["arm_qpos"],
                        RESET_CMD[self.object_type],
                        atol=0.02,
                    )
                    if reset_done:
                        break
                    print("Waiting for arm to reset...")
                    time.sleep(POLICY_CONTROL_PERIOD)

            # Initialize segmentation for all cameras
            self.camera_data = self.get_camera_data()
            rgb_frames = {
                camera_name: self.camera_data[camera_id]["color"][..., ::-1]
                for camera_id, camera_name in enumerate(self.camera_names)
            }

            # Initialize segmentation (blocking operation to ensure first masks are ready)
            self.sam_manager.init_all_seg(rgb_frames)
            print("Initialized async SAM segmentation for all cameras")
            self.sam_initialized = True

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.arm.execute_action(action)  # Non-blocking
        print(f"Arm step: {action}")

    def close(self):
        super().close()
        self.arm.close()


if __name__ == "__main__":
    import time

    import numpy as np

    from constants import POLICY_CONTROL_PERIOD

    env = ArmEnv(
        camera_names=["wrist", "wrist2"],  # Must match what's in CAMERA_CONFIGS
        sam=True,
        object_type="long",
        sam_model_type="t",  # Use a fast model for testing
    )
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    "base_pose": 0.1 * np.random.rand(3) - 0.05,
                    "arm_pos": 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    "arm_quat": np.random.rand(4),
                    "gripper_pos": np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print("Observation keys and types:")
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k}: np.ndarray, shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, dict) and "binary_mask" in v:
                        print(
                            f"  {k}: SAM Mask dict, binary_mask shape={v['binary_mask'].shape}, age_ms={v['age_ms']}"
                        )
                    else:
                        print(f"  {k}: {type(v)}")

                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
