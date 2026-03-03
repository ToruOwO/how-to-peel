import argparse
import sys
import time
from contextlib import ExitStack, contextmanager
from itertools import count
from multiprocessing.managers import SharedMemoryManager

from arm_env import ArmEnv, MockArmEnv
from constants import CAMERA_CONFIGS, POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from single_realsense import SingleRealsense


@contextmanager
def activate_cameras(cameras):
    """
    Context manager to activate multiple cameras.

    Args:
        cameras: List of camera objects that have an activate() method

    Yields:
        The list of cameras (for convenience)
    """
    with ExitStack() as stack:
        # Activate all cameras
        for camera in cameras:
            stack.enter_context(camera.activate())
        yield cameras


def should_save_episode(writer) -> int:
    if len(writer) == 0:
        print("Discarding empty episode")
        return 0

    # Prompt user whether to save episode with score
    while True:
        user_input = input("Episode score? (0-9)?").strip()
        try:
            user_input = int(user_input)
        except ValueError:
            print("Invalid response", user_input)
            continue

        print(f"User input: {user_input}")
        if 0 <= user_input <= 9:
            if user_input == 0:
                print("Discarding episode")
                return 0
            else:
                print("Saving episode")
                return user_input
        else:
            print("Invalid response")
            continue


def run_episode(env, policy, writer=None):
    # Wait for user to press "Start episode"
    policy.reset()

    print("Resetting env...")
    reset_arm = input("Reset arm (y/n)? Ready to start new episode ").strip().lower()
    env.reset(reset_arm=(reset_arm == "y"))
    print("Env has been reset")

    # Get current ee pose
    q_init = env.get_obs()["arm_qpos"]
    print(f"q_init: {q_init}")
    policy.init(q_init)

    print("Starting new episode")

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        obs = env.get_obs()

        # Get action
        action = policy.step(obs)

        # No action if teleop not enabled
        if action is None:
            continue

        # Execute valid action on robot
        if isinstance(action, dict):
            print("\nvalid action: ", action)
            env.step(action)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(obs, action)

        # Episode ended
        elif not episode_ended and action == "end_episode":
            episode_ended = True
            print("Episode ended")

            if writer is not None:
                score = should_save_episode(writer)
                if score > 0:
                    # Save to disk in background thread
                    writer.flush_async(score)

            print('Teleop is now active. Press "Reset" when ready to proceed.')

        # Ready for env reset
        elif action == "reset_env":
            break

    if writer is not None:
        # Wait for writer to finish saving to disk
        writer.wait_for_flush()


def main(args):
    if args.test:
        from policies import (
            KeyboardTeleopPolicy,
            PlannerPolicy,
            ReplayPolicy,
            ScriptedTrajectoryPolicy,
            SpacemouseTeleopPolicy,
            TorchPolicy,
        )

        # no real camera or robot needed
        camera_names = ["wrist", "wrist2"]
        env = MockArmEnv(
            cameras=None,
            camera_names=camera_names,
            sam=args.sam,
            object_type=args.object_type,
        )
        control_mode = "eef" if args.eef else "world"

        if args.ckpt_dir:
            policy = TorchPolicy(
                env,
                args.ckpt_dir,
                control_mode=control_mode,
                use_default_rot=(args.object_type == "short"),
                res_ckpt_path=args.res_ckpt_dir
                if hasattr(args, "res_ckpt_dir")
                else None,
            )
        elif args.replay:
            policy = ReplayPolicy(env, args.replay, control_mode=control_mode)
        elif args.planner:
            policy = PlannerPolicy(env, control_mode=control_mode)
        elif args.keyboard:
            policy = KeyboardTeleopPolicy(env, control_mode=control_mode)
        else:
            policy = SpacemouseTeleopPolicy(
                env,
                control_mode=control_mode,
                use_default_rot=(args.object_type == "short"),
            )

        try:
            while True:
                writer = EpisodeWriter(args, camera_names) if args.save else None
                run_episode(env, policy, writer)
        finally:
            env.close()
            sys.exit(0)
    else:
        with SharedMemoryManager() as shm_manager:
            camera_names = ["wrist", "wrist2"]
            cameras = []
            for camera_name in camera_names:
                camera = SingleRealsense(
                    shm_manager=shm_manager,
                    serial_number=CAMERA_CONFIGS[camera_name]["serial_number"],
                    product_line=CAMERA_CONFIGS[camera_name]["product_line"],
                    depth_resolution=CAMERA_CONFIGS[camera_name]["depth_resolution"],
                    color_resolution=CAMERA_CONFIGS[camera_name]["color_resolution"],
                    capture_fps=CAMERA_CONFIGS[camera_name]["capture_fps"],
                    enable_color=CAMERA_CONFIGS[camera_name]["enable_color"],
                    enable_depth=CAMERA_CONFIGS[camera_name]["enable_depth"],
                    verbose=False,
                )
                cameras.append(camera)

            # Activate all cameras using our helper function
            with activate_cameras(cameras) as active_cameras:
                if args.test_camera:
                    env = MockArmEnv(
                        cameras=active_cameras,
                        camera_names=camera_names,
                        sam=args.sam,
                        object_type=args.object_type,
                        sam_model_type=args.sam_model_type,
                        sam_fps=args.sam_fps,
                    )
                else:
                    # Create env
                    env = ArmEnv(
                        cameras=active_cameras,
                        camera_names=camera_names,
                        sam=args.sam,
                        object_type=args.object_type,
                        sam_model_type=args.sam_model_type,
                        sam_fps=args.sam_fps,
                    )

                control_mode = "eef" if args.eef else "world"
                print(f"Using control mode: {control_mode}")

                from policies import (
                    KeyboardTeleopPolicy,
                    PlannerPolicy,
                    ReplayPolicy,
                    ScriptedTrajectoryPolicy,
                    SpacemouseTeleopPolicy,
                    TorchPolicy,
                )

                if args.ckpt_dir:
                    policy = TorchPolicy(
                        env,
                        args.ckpt_dir,
                        control_mode=control_mode,
                        use_default_rot=(args.object_type == "short"),
                        use_ema=args.ema,
                        res_ckpt_path=args.res_ckpt_dir
                        if hasattr(args, "res_ckpt_dir")
                        else None,
                    )
                elif args.replay:
                    policy = ReplayPolicy(env, args.replay, control_mode=control_mode)
                elif args.scripted:
                    policy = ScriptedTrajectoryPolicy(env)
                elif args.planner:
                    policy = PlannerPolicy(env, control_mode=control_mode)
                elif args.keyboard:
                    policy = KeyboardTeleopPolicy(env, control_mode=control_mode)
                else:
                    policy = SpacemouseTeleopPolicy(
                        env,
                        control_mode=control_mode,
                        use_default_rot=(args.object_type == "short"),
                    )

                try:
                    while True:
                        writer = (
                            EpisodeWriter(args, camera_names) if args.save else None
                        )
                        run_episode(env, policy, writer)
                finally:
                    env.close()
                    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-camera", action="store_true")

    # controller configs
    parser.add_argument("--pos", action="store_true", help="Use position control")

    # teleop configs
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--eef", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--sam-model-type", type=str, default="b+")
    parser.add_argument("--sam-fps", type=float, default=3)

    # policy configs
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument("--replay", type=str, default="")
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--res-ckpt-dir", type=str, default="")
    parser.add_argument("--object-type", type=str, default="long")
    parser.add_argument("--ema", action="store_true")

    # save configs
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", default="data/demos")
    main(parser.parse_args())
