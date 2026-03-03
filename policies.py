import csv
import time

import numpy as np
from diffusion_policy.export import build_diffusion_model_from_ckpt  # NOTE: this API is not included in this code base and must be custom defined
from residual_policy_with_iql.export import build_residual_model_from_ckpt
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from cameras.calibration_utils import load_camera_info
from constants import CAMERA_CONFIGS, DATASET_VERSION, POLICY_CONTROL_FREQ
from episode_storage import EpisodeReader
from keyboard_device import KeyboardUI
from sam_camera import Segment
from spacemouse import SpaceMouse
from traj_utils import TrajectoryProcessor, create_point_cloud, o3d_pcd_with_normals


def get_world_target(pos, quat, delta_pos, delta_rot, frame="world"):
    """

    frame: in which frame the delta_pos and delta_rot are given

    return: new_pos, new_quat in world frame
    """
    rot_mat = R.from_quat(quat).as_matrix()
    delta_rot_matrix = R.from_euler("xyz", delta_rot).as_matrix()

    current_transform = np.eye(4)
    current_transform[:3, :3] = rot_mat
    current_transform[:3, 3] = pos  # T_base_eef

    if frame == "eef":
        next_transform_in_eef = np.eye(4)  # T_eef_nexteef
        next_transform_in_eef[:3, 3] = delta_pos
        next_transform_in_eef[:3, :3] = delta_rot_matrix
        result = current_transform @ next_transform_in_eef  # T_base_nexteef

    elif frame == "world":
        result = np.array(current_transform)
        result[:3, 3] += delta_pos
        result[:3, :3] = delta_rot_matrix @ result[:3, :3]

    else:
        raise ValueError(f"Invalid frame: {frame}")

    new_pos = result[:3, 3]  # [3,]
    new_quat = R.from_matrix(result[:3, :3]).as_quat()  # [4,]

    cprint(f"Current pose: {pos}, {(quat)}", "green")
    cprint(f"New pose: {new_pos}, {(new_quat)}", "green")
    return new_pos, new_quat


def get_local_delta(pos, quat, next_pos, next_quat, delta_frame="world"):
    """
    pos, quat, next_pos, next_quat: EEF pose in world frame
    Get the local delta_pos and delta_rot in the delta_frame.
    """
    R_cur = R.from_quat(quat)
    R_nxt = R.from_quat(next_quat)
    R_delta = R_nxt * R_cur.inv()
    dpos_world = np.asarray(next_pos) - np.asarray(pos)

    if delta_frame == "world":
        delta_pos = dpos_world
        delta_rot = R_delta.as_euler("xyz", degrees=False)

    elif delta_frame == "eef":
        # express translation in EEF frame
        delta_pos = R_cur.inv().apply(dpos_world)

        # TODO: is this correct?
        delta_rot = R_delta.as_euler("xyz", degrees=False)

    else:
        raise ValueError(f"Invalid delta_frame: {delta_frame}")

    return delta_pos, delta_rot


class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError


class TeleopPolicy(Policy):
    def __init__(self, env, control_mode="world"):
        self.env = env
        self.control_mode = control_mode

        self.q_init = None
        self.q_current = None

        # NOTE: these are in world frame!
        self.pos_init = None
        self.quat_init = None
        self.pos_current = None
        self.quat_current = None

        self.xyz_speed = min(0.1 / POLICY_CONTROL_FREQ, 0.02)
        self.rot_speed = min(np.deg2rad(6) / POLICY_CONTROL_FREQ, np.deg2rad(1.2))

    def init(self, q_init):
        self.q_init = q_init
        self.pos_init, self.quat_init = self.env.ik_solver.run_fk(q_init)
        self.q_current = q_init.copy()
        self.pos_current = self.pos_init.copy()
        self.quat_current = self.quat_init.copy()

    def q_update(self, qpos):
        self.q_current = qpos
        self.pos_current, self.quat_current = self.env.ik_solver.run_fk(qpos)

    def reset(self):
        pass


class TorchPolicy(TeleopPolicy):
    def __init__(
        self,
        env,
        ckpt_path,
        control_mode="world",
        use_default_rot=False,
        use_ema=False,
        res_ckpt_path=None,
    ):
        super().__init__(env, control_mode)

        # init model agent
        self.model = build_diffusion_model_from_ckpt(ckpt_path, ema=use_ema)
        self.residual_model = (
            build_residual_model_from_ckpt(res_ckpt_path) if res_ckpt_path else None
        )
        self.lam = 0.5  # residual scaling factor
        print("Diffusion model and residual model loaded.")

        self.use_default_rot = use_default_rot

        # for takeover
        self.teleop_mode = False
        self.teleop_policy = SpacemouseTeleopPolicy(
            env, control_mode=control_mode, use_default_rot=use_default_rot
        )

        # to control play/pause
        self.keyboard_ui = KeyboardUI()
        self.keyboard_ui.start()

    def reset(self):
        self.teleop_mode = False

    def preprocess_obs(self, obs):
        rgb = obs["wrist_rgb"]

        depth = obs["wrist_depth"]

        if "mask" in obs:
            mask = obs["mask"]["binary_mask"]

        processed_obs = {
            "depth": depth.astype(np.float32),  # [H, W]
            "rgb": rgb.astype(np.uint8),  # [H, W, 3]
            "mask": mask.astype(np.float32),  # [H, W]
            "force": obs["force"].astype(np.float32),  # [6,]
        }
        return processed_obs

    def preprocess_obs_v3(self, obs):
        processed_obs = {"observation": obs}
        return processed_obs

    def step(self, obs):
        print("\n--- TorchPolicy Step ---")
        cmd = list(self.keyboard_ui.pressed_keys)

        if "t" in cmd:
            self.teleop_mode = not self.teleop_mode
            if self.teleop_mode:
                self.teleop_policy.init(obs["arm_qpos"])
            return None

        if self.teleop_mode:
            cprint("Teleop mode", "red")
            return self.teleop_policy.step(obs)

        if "q" in cmd:
            return "end_episode"
        if "r" in cmd:
            return "reset_env"
        if "p" not in cmd:
            cprint("\nPress 'p' to pause, otherwise play", "green")
            if DATASET_VERSION < 3:
                processed_obs = self.preprocess_obs(obs)
            else:
                processed_obs = self.preprocess_obs_v3(obs)

            if self.residual_model is None:
                action = self.model.predict(processed_obs)
            else:
                print("Using residual model.")
                action_norm, z_cond = self.model.predict_action_normalized(
                    processed_obs
                )
                print("action_norm:", action_norm)
                delta_action_norm = self.residual_model.forward(z_cond)
                delta_action_norm = delta_action_norm.detach().cpu().numpy()[0]
                print("delta action_norm:", delta_action_norm)

                action = np.clip(action_norm + self.lam * delta_action_norm, -1.0, 1.0)
                action = self.model.data_processor.denormalize_output(action)
                print("Final action:", action)

            delta_pos, delta_rot = action[:3], action[3:]

            if not self.use_default_rot:
                delta_rot[1] = 0.0  # HACK: follow teleop

            new_pos, new_quat = get_world_target(
                self.pos_current.copy(),
                self.quat_current.copy(),
                delta_pos,
                delta_rot,
                frame=self.control_mode,
            )
            qpos = self.env.ik_solver.solve(new_pos, new_quat, self.q_current)
            self.q_update(qpos)
            return {
                "arm_qpos": qpos,
                "eef_delta_pos": delta_pos,
                "eef_delta_rot": delta_rot,
            }
        else:
            return None


class ReplayPolicy(TeleopPolicy):
    def __init__(self, env, episode_dir, control_mode="world"):
        super().__init__(env, control_mode)
        self.reader = EpisodeReader(episode_dir)
        self.step_idx = 0

    def step(self, obs):
        if self.step_idx >= len(self.reader):
            return "end_episode"
        else:
            action = self.reader.actions[self.step_idx]
            self.step_idx += 1
            return action


class CalibrateReplayPolicy(TeleopPolicy):
    def __init__(self, env, csv_path, control_mode="world"):
        super().__init__(env, control_mode)
        self.csv_path = csv_path
        self.qpos_list = []
        self.action_repeat = 3
        self.step_idx = 0

    def init(self, q_init):
        super().init(q_init)
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                qpos = [
                    float(row[8]),
                    float(row[9]),
                    float(row[10]),
                    float(row[11]),
                    float(row[12]),
                    float(row[13]),
                    float(row[14]),
                ]
                self.qpos_list.extend([qpos] * self.action_repeat)

    def step(self, obs):
        if self.step_idx >= len(self.qpos_list):
            return "end_episode"
        else:
            qpos = self.qpos_list[self.step_idx]
            pos, quat = self.env.ik_solver.run_fk(qpos)

            self.step_idx += 1
            return {
                "arm_qpos": qpos,
                "arm_pos": pos,
                "arm_quat": quat,
            }


class SpacemouseTeleopPolicy(TeleopPolicy):
    def __init__(
        self,
        env,
        control_mode="world",
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        use_default_rot=False,
    ):
        super().__init__(env, control_mode)
        self.sm = SpaceMouse(
            pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity
        )

        self.xyz_speed = 0.005
        self.rot_speed = np.deg2rad(0.3)
        self.use_default_rot = use_default_rot

    def init(self, q_init):
        super().init(q_init)
        self.sm.start_control()

    def step(self, obs):
        time.sleep(0.002)
        button_state = self.sm.button_state
        # cprint(f"Buttons: {button_state}", "yellow")
        if button_state[0]:  # left pressed
            self.sm.reset_button_state()
            return "end_episode"
        elif button_state[1]:  # right pressed
            self.sm.reset_button_state()
            return "reset_env"

        cmd = self.sm.control  # [x, y, z, roll, pitch, yaw]

        # return None if all close to zero
        if np.all(np.abs(cmd) < 0.001):
            return None

        cprint(str(cmd), "yellow")

        if self.control_mode == "eef":
            cmd[0] = -cmd[0]  # x
            cmd[1] = -cmd[1]  # y
            cmd[3] = -cmd[3]  # pitch
            cmd[4] = -cmd[4]  # roll

        delta_pos = np.array([cmd[0], cmd[1], cmd[2]]) * self.xyz_speed

        if self.use_default_rot:
            delta_rot = np.array([cmd[4], cmd[3], -cmd[5]]) * self.rot_speed
        else:
            delta_rot = np.array([cmd[4], 0.0, -cmd[5] * 0.5]) * self.rot_speed  # HACK

        new_pos, new_quat = get_world_target(
            self.pos_current.copy(),
            self.quat_current.copy(),
            delta_pos,
            delta_rot,
            frame=self.control_mode,
        )

        qpos = self.env.ik_solver.solve(new_pos, new_quat, self.q_current)
        self.q_update(qpos)
        return {
            "arm_qpos": qpos,
            "eef_delta_pos": delta_pos,  # in meters
            "eef_delta_rot": delta_rot,  # rpy in radians
        }


class KeyboardTeleopPolicy(TeleopPolicy):
    def __init__(self, env, control_mode="world"):
        super().__init__(env, control_mode)

        self.keyboard_ui = KeyboardUI()
        self.keyboard_ui.start()

    def step(self, obs):
        cprint("\n[Command Options]", "yellow")
        cprint("w: x+    s: x-", "yellow")
        cprint("a: y+    d: y-", "yellow")
        cprint("q: z+    e: z-", "yellow")
        cprint("z: roll+    x: roll-", "yellow")
        cprint("c: pitch+    v: pitch-", "yellow")
        cprint("n: yaw+    b: yaw-", "yellow")
        # cmd = input("\033[1;35mInput command: \033[0m")

        cmd = list(self.keyboard_ui.pressed_keys)
        print("Keys", cmd)
        if len(cmd) == 0:
            return None

        cmd = cmd[0]

        delta_pos = [0.0, 0.0, 0.0]
        delta_rot = [0.0, 0.0, 0.0]

        if cmd == "w":
            delta_pos[0] += self.xyz_speed  # Forward (x)
        elif cmd == "s":
            delta_pos[0] -= self.xyz_speed  # Backward (x)
        elif cmd == "a":
            delta_pos[1] += self.xyz_speed  # Left (y)
        elif cmd == "d":
            delta_pos[1] -= self.xyz_speed  # Right (y)
        elif cmd == "q":
            delta_pos[2] += self.xyz_speed  # Up (z)
        elif cmd == "e":
            delta_pos[2] -= self.xyz_speed  # Down (z)

        elif cmd == "z":
            delta_rot[0] += self.rot_speed  # Roll (X)
        elif cmd == "x":
            delta_rot[0] -= self.rot_speed  # Reverse Roll (X)
        elif cmd == "c":
            delta_rot[1] += self.rot_speed  # Pitch (Y)
        elif cmd == "v":
            delta_rot[1] -= self.rot_speed  # Reverse Pitch (Y)
        elif cmd == "b":
            delta_rot[2] += self.rot_speed  # Yaw (Z)
        elif cmd == "n":
            delta_rot[2] -= self.rot_speed  # Reverse Yaw (Z)

        new_pos, new_quat = get_world_target(
            self.pos_current.copy(),
            self.quat_current.copy(),
            delta_pos,
            delta_rot,
            frame=self.control_mode,
        )

        qpos = self.env.ik_solver.solve(new_pos, new_quat, self.q_current)

        self.q_update(qpos)
        return {
            "arm_qpos": qpos,
            "eef_delta_pos": delta_pos,
            "eef_delta_rot": delta_rot,
        }


class ScriptedTrajectoryPolicy(Policy):
    def __init__(self, env, traj_type="world"):
        self.count = 0
        self.env = env
        self.q_init = None
        self.pos_init = None
        self.quat_init = None

        self.traj_type = traj_type

        if traj_type == "eef":
            delta = 0.002
            self.delta_list = (
                [delta * (i + 1) for i in range(20)]
                + [delta * (20 - i) for i in range(1, 20)]
                + [0.0]
                + [-delta * (i + 1) for i in range(20)]
                + [-delta * (20 - i) for i in range(1, 20)]
                + [0.0]
            )
            self.delta_idx = [1, 2]
        else:
            # traj type is joint
            delta = np.deg2rad(0.5)
            self.delta_list = (
                [delta * (i + 1) for i in range(20)]
                + [delta * (20 - i) for i in range(1, 20)]
                + [0.0]
                + [-delta * (i + 1) for i in range(20)]
                + [-delta * (20 - i) for i in range(1, 20)]
                + [0.0]
            )
            self.delta_idx = [0, 1, 2, 3, 4, 5, 6]

    def init(self, q_init):
        self.q_init = q_init
        self.pos_init, self.quat_init = self.env.ik_solver.run_fk(q_init)

    def reset(self):
        self.count = 0

    def step(self, obs):
        # input("Press enter to continue")
        step = self.count % (len(self.delta_list) * len(self.delta_idx))
        curr_delta_idx = step // len(self.delta_list)
        curr_delta = self.delta_list[step % len(self.delta_list)]

        if self.traj_type == "eef":
            next_pos = self.pos_init.copy()
            next_pos[self.delta_idx[curr_delta_idx]] += curr_delta
            print(next_pos)
            qpos = self.env.ik_solver.solve(next_pos, self.quat_init, self.q_init)
        else:
            next_qpos = self.q_init.copy()
            next_qpos[self.delta_idx[curr_delta_idx]] += curr_delta
            qpos = next_qpos

        print(f"curr_delta_idx: {curr_delta_idx}, curr_delta: {curr_delta}")

        action = {"arm_qpos": qpos}

        print(self.count)

        self.count += 1

        return action


class PlannerPolicy(TeleopPolicy):
    def __init__(self, env, control_mode="world"):
        super().__init__(env, control_mode)
        self.sam = Segment(model_type="b+", show_live=False, object_names=["object"])
        self.traj_processor = None

        self.d = 0.005
        self.delta_to_normal_range = [75, 75]

    def _init_traj_processor(self):
        for i in range(3):
            time.sleep(1)
            print("Waiting for arm to be ready...", i + 1)

        # load rgb, depth, intrinsics from saved data
        camera_info = load_camera_info(CAMERA_CONFIGS["lidar"]["serial_number"])
        rgb = camera_info["color"][:, :, ::-1]  # [1080, 1920, 3] uint8
        depth = camera_info["depth"]  # [1080, 1920] uint16
        intrinsics = camera_info["intrinsics"]
        extrinsics_matrix = camera_info["extrinsics"]

        camera_location = extrinsics_matrix[:3, 3].tolist()

        self.sam.select_prompts(rgb)
        masks = self.sam.step(rgb)[0]  # dict of object names to masks
        segment_mask = masks["object"]  # (H, W) binary mask

        pointcloud_np = create_point_cloud(
            depth,
            rgb,
            intrinsics,
            extrinsics_matrix,
            segment_mask,
            apply_dbscan=True,
            dbscan_eps=0.006,
            dbscan_min_samples=200,
        )
        pcd_o3d = o3d_pcd_with_normals(
            pointcloud_np,
            normal_radius=0.01,
            normal_max_nn=2000,
            camera_location=camera_location,
        )
        self.traj_processor = TrajectoryProcessor(
            pcd_o3d,
            traj_reserve_ratio=0.8,
            fit_x_window=[0.3, 0.5],
        )
        self.traj_processor.visualize_ref_traj_tangents(num_samples=20)

    def init(self, q_init):
        super().init(q_init)
        self._init_traj_processor()
        self.next_xyz, self.next_forward_dir = self.traj_processor.get_start_point(
            delta_to_normal=75
        )
        self.next_qpos = self.extract_knife_pose(self.next_xyz, self.next_forward_dir)

    def compute_ee_rotation(self, forward_dir, normal):
        y_des = forward_dir / np.linalg.norm(forward_dir)
        z_init = normal

        z_des = z_init - np.dot(z_init, y_des) * y_des
        z_norm = np.linalg.norm(z_des)
        z_des = z_des / z_norm

        x_des = np.cross(y_des, z_des)
        x_des = x_des / np.linalg.norm(x_des)

        R = np.stack([x_des, y_des, z_des], axis=1)
        return np.array(R)

    def extract_knife_pose(self, point_xyz, point_forward_dir):
        normal = -self.traj_processor.get_normal(point_xyz)
        ee_rot_mat = self.compute_ee_rotation(point_forward_dir, normal)
        ee_quat = R.from_matrix(ee_rot_mat).as_quat()
        qpos = self.env.ik_solver.solve(point_xyz, ee_quat, self.q_current)
        return qpos

    @staticmethod
    def _angle_err(a, b):
        # shortest signed diff in (-pi, pi]
        return ((a - b + np.pi) % (2 * np.pi)) - np.pi

    def step(self, obs):
        # compare diff with observed qpos
        qpos_diff = np.abs(self._angle_err(self.next_qpos, obs["arm_qpos"]))
        qpos_diff_max = qpos_diff.max()
        print(f"qpos_diff: {np.rad2deg(qpos_diff)}, {np.rad2deg(qpos_diff_max)}")

        if qpos_diff_max > 0.05:
            print("qpos_diff is too large")
            xyz, forward_dir = self.next_xyz, self.next_forward_dir
            qpos = self.next_qpos
        else:
            # continue to the next point
            print("sampling next point")
            xyz, forward_dir = self.traj_processor.sample_next_point(
                self.next_xyz,
                self.next_forward_dir,
                self.d,
                self.delta_to_normal_range,
            )
            self.next_xyz, self.next_forward_dir = xyz, forward_dir
            qpos = self.extract_knife_pose(xyz, forward_dir)
            self.next_qpos = qpos
            input("Press Enter to continue")

        print(f"xyz: {xyz}, forward_dir: {forward_dir}")

        # calculate the corresponding delta_pos and delta_rot
        next_pos, next_rot = self.env.ik_solver.run_fk(qpos)
        delta_pos, delta_rot = get_local_delta(
            self.pos_current,
            self.quat_current,
            next_pos,
            next_rot,
            delta_frame=self.control_mode,
        )

        # update q_current, pos_current, quat_current for next step
        self.q_update(qpos)

        return {
            "arm_qpos": qpos,
            "eef_delta_pos": delta_pos,
            "eef_delta_rot": delta_rot,
        }


if __name__ == "__main__":
    from arm_env import MockArmEnv

    env = MockArmEnv()
    policy = TorchPolicy(env, "./ckpts/test")
    q_init = env.get_obs()["arm_qpos"]
    policy.init(q_init)
    obs = env.get_obs()
    action = policy.step(obs)
    print(action)
