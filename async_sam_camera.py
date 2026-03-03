import multiprocessing as _mp
import queue
import time
from typing import Dict, List

import numpy as np
from termcolor import cprint

from sam_camera import Segment


_SAM_CTX = _mp.get_context("spawn")

class SamWorker:
    """
    A worker process to run SAM segmentation for a single camera.
    """

    def __init__(
        self, camera_name: str, model_type: str, object_names: List[str], device: str
    ):
        self.camera_name = camera_name
        self.segmentor = Segment(
            model_type=model_type,
            show_live=False,  # Live display is handled by the main process if needed
            object_names=object_names,
            device=device,
        )
        self.segmentor_initialized = False

    def init_seg(self, init_rgb: np.ndarray):
        """
        Initializes the segmentor with prompts. This is a blocking call.
        """
        cprint(f"[SAM Worker {self.camera_name}] Initializing segmentation...", "cyan")
        # Set prompts based on the initial prompts provided
        self.segmentor.select_prompts(init_rgb)

        # The first `step` call after setting prompts will trigger the internal `initialize`
        # of the Segment class, which loads the first frame and processes prompts.
        # We perform a dummy step to get the initial masks.
        masks, _ = self.segmentor.step(init_rgb)  # This call is blocking

        self.segmentor_initialized = True
        cprint(f"[SAM Worker {self.camera_name}] Segmentation initialized.", "green")
        return masks

    def process_frame(self, rgb_frame: np.ndarray) -> Dict:
        """
        Processes a single RGB frame to get masks.
        """
        if not self.segmentor_initialized:
            cprint(
                f"[SAM Worker {self.camera_name}] Warning: Attempted to process frame before initialization.",
                "yellow",
            )
            return {
                "binary_mask": np.zeros(rgb_frame.shape[:2], dtype=np.uint8),
                "masks": {},
                "age_ms": float("inf"),  # Indicate invalid mask
            }

        start_time = time.perf_counter()
        masks, _ = self.segmentor.step(rgb_frame)
        end_time = time.perf_counter()

        binary_mask = Segment.get_mask_union(masks)

        return {
            "binary_mask": binary_mask,
            "masks": masks,
            "age_ms": (end_time - start_time)
            * 1000,  # Time taken to process this frame
        }

    def reset(self):
        """
        Resets the worker's segmentor.
        """
        cprint(f"[SAM Worker {self.camera_name}] Resetting segmentor...", "cyan")
        self.segmentor.reset()
        self.segmentor_initialized = False


def _sam_worker_fn(
    camera_name: str,
    model_type: str,
    object_names: List[str],
    device: str,
    input_queue: _mp.Queue,
    output_queue: _mp.Queue,
    control_queue: _mp.Queue,
):
    """
    The target function for the SAM worker process.
    """
    # <--- IMPORT TORCH HERE, INSIDE THE WORKER FUNCTION
    import torch

    # Re-check CUDA availability in the worker process
    if "cuda" in device and not torch.cuda.is_available():
        cprint(
            f"[SAM Worker {camera_name}] CUDA not available on device {device}, falling back to CPU.",
            "red",
        )
        device = "cpu"
    else:
        # Ensure correct device is set for the worker
        if "cuda" in device:
            try:
                device_id = int(device.split(":")[-1])
                torch.cuda.set_device(device_id)
                cprint(
                    f"[SAM Worker {camera_name}] Set CUDA device to {device_id}.",
                    "green",
                )
            except Exception as e:
                cprint(
                    f"[SAM Worker {camera_name}] Could not set CUDA device {device}: {e}, falling back to CPU.",
                    "red",
                )
                device = "cpu"

    worker = SamWorker(camera_name, model_type, object_names, device)
    running = True

    while running:
        # Check for control commands first
        try:
            command, data = control_queue.get(
                timeout=0.01
            )  # Small timeout to allow processing
            if command == "INIT_SEG":
                init_rgb = data["init_rgb"]
                masks = worker.init_seg(init_rgb)
                output_queue.put(
                    {"camera_name": camera_name, "masks": masks, "age_ms": 0}
                )
            elif command == "RESET":
                worker.reset()
            elif command == "STOP":
                running = False
        except queue.Empty:
            pass  # No control command, proceed to process frames

        # Process frames if available
        try:
            rgb_frame_id, rgb_frame = input_queue.get(timeout=0.01)  # Small timeout
            processed_data = worker.process_frame(rgb_frame)
            processed_data["frame_id"] = (
                rgb_frame_id  # Keep track of which frame this mask belongs to
            )
            output_queue.put({"camera_name": camera_name, **processed_data})
        except queue.Empty:
            pass  # No frames to process
        except Exception as e:
            cprint(f"[SAM Worker {camera_name}] Error processing frame: {e}", "red")
            # If a worker encounters an error, it's safer to stop it to prevent deadlocks
            running = False  # <--- ADDED: Stop worker on error


class AsyncSamCameraManager:
    """
    Manages multiple SAM worker processes for asynchronous segmentation.
    """

    def __init__(
        self,
        cameras: List,  # List of camera objects (e.g., SingleRealsense)
        camera_names: List[str],
        sam_fps: float,
        model_type: str,
        device_base: str = "cuda",
        object_names: List[str] = ["knife", "object"],
    ):
        self.cameras = cameras
        self.camera_names = camera_names
        self.sam_fps = sam_fps
        self.model_type = model_type
        self.device_base = device_base
        self.object_names = object_names

        self.input_queues: Dict[str, _mp.Queue] = {}
        self.output_queue: _mp.Queue = _SAM_CTX.Queue()
        self.control_queues: Dict[str, _mp.Queue] = {}
        self.processes: Dict[str, _mp.Process] = {}
        self.latest_masks: Dict[str, Dict] = {
            name: {
                "binary_mask": np.zeros(
                    (480, 640), dtype=np.uint8
                ),  # Default empty mask
                "masks": {},
                "age_ms": float("inf"),  # Initially infinitely old
                "frame_id": -1,
            }
            for name in camera_names
        }
        self.current_frame_id = 0
        self.started = False
        self.last_submit_time = 0

        # Assign devices round-robin if multiple CUDA devices exist

        # NOTE: hardcoded to avoid torch import in parent process
        num_cuda_devices = 2
        self.worker_devices = {}
        for i, cam_name in enumerate(camera_names):
            if "cuda" in device_base and num_cuda_devices > 0:
                self.worker_devices[cam_name] = f"cuda:{i % num_cuda_devices}"
            else:
                self.worker_devices[cam_name] = "cpu"
            cprint(
                f"Assigned {cam_name} to device: {self.worker_devices[cam_name]}",
                "yellow",
            )

    def start_all(self):
        """
        Starts all SAM worker processes.
        """
        if self.started:
            cprint("[AsyncSAM] Workers already started.", "yellow")
            return

        cprint("[AsyncSAM] Starting SAM worker processes...", "cyan")
        for camera_name in self.camera_names:
            input_q = _SAM_CTX.Queue(maxsize=2)  # Buffer for 2 frames
            control_q = _SAM_CTX.Queue(maxsize=1)
            self.input_queues[camera_name] = input_q
            self.control_queues[camera_name] = control_q

            p = _SAM_CTX.Process(
                target=_sam_worker_fn,
                args=(
                    camera_name,
                    self.model_type,
                    self.object_names,
                    self.worker_devices[camera_name],
                    input_q,
                    self.output_queue,
                    control_q,
                ),
            )
            self.processes[camera_name] = p
            p.daemon = (
                True  # Allow main program to exit even if workers are still running
            )
            p.start()
            cprint(f"  Worker for {camera_name} started (PID: {p.pid})", "green")
        self.started = True
        cprint("[AsyncSAM] All SAM worker processes started.", "green")

    def stop_all(self):
        """
        Stops all SAM worker processes.
        """
        if not self.started:
            cprint("[AsyncSAM] Workers not started, nothing to stop.", "yellow")
            return

        cprint("[AsyncSAM] Stopping SAM worker processes...", "cyan")
        for camera_name, control_q in self.control_queues.items():
            try:
                control_q.put(("STOP", None))
            except Exception:
                pass

        # Join processes
        for camera_name, p in self.processes.items():
            p.join(timeout=5)
            if p.is_alive():
                cprint(
                    f"  Worker for {camera_name} did not terminate, forcing kill.",
                    "red",
                )
                p.terminate()
                p.join(timeout=2)

        # Close & join all queues to stop feeder threads cleanly
        for q in self.input_queues.values():
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        try:
            self.output_queue.close()
            self.output_queue.join_thread()
        except Exception:
            pass
        for q in self.control_queues.values():
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        self.input_queues.clear()
        # Recreate a fresh output_queue for next start, if any
        self.output_queue = _SAM_CTX.Queue()
        self.control_queues.clear()
        self.processes.clear()
        self.started = False
        cprint("[AsyncSAM] All SAM worker processes stopped.", "green")

    def init_all_seg(self, rgb_frames: Dict[str, np.ndarray]):
        """
        Initializes segmentation for all cameras by sending blocking commands to workers.
        This will block until all workers have reported back their initial masks.
        """
        if not self.started:
            self.start_all()  # Ensure workers are running before initialization

        cprint("[AsyncSAM] Initializing segmentation in all workers...", "cyan")
        # For each camera, send an INIT_SEG command
        for cam_name in self.camera_names:
            if cam_name not in rgb_frames:
                raise ValueError(
                    f"Missing RGB frame for camera {cam_name} during init_all_seg."
                )

            self.control_queues[cam_name].put(
                (
                    "INIT_SEG",
                    {
                        "init_rgb": rgb_frames[cam_name],
                    },
                )
            )

        # Wait for all workers to send back their initial masks
        initialized_count = 0
        while initialized_count < len(self.camera_names):
            try:
                result = self.output_queue.get(
                    timeout=30
                )  # Long timeout for interactive selection
                cam_name = result["camera_name"]
                if "masks" in result:  # This is a result from INIT_SEG
                    self.latest_masks[cam_name] = {
                        "binary_mask": Segment.get_mask_union(result["masks"]),
                        "masks": result["masks"],
                        "age_ms": 0,  # Fresh mask
                        "frame_id": self.current_frame_id,
                    }
                    cprint(f"  Received initial masks for {cam_name}", "green")
                    initialized_count += 1
            except queue.Empty:
                cprint(
                    "[AsyncSAM] Waiting for SAM workers to initialize segmentation...",
                    "yellow",
                )
                # Check if workers are still alive to avoid infinite loop on crashed workers
                if not all(p.is_alive() for p in self.processes.values()):
                    cprint(
                        "[AsyncSAM] One or more SAM workers crashed during initialization!",
                        "red",
                    )
                    raise RuntimeError(
                        "SAM worker process crashed during initialization."
                    )
                time.sleep(
                    1
                )  # Wait longer if workers are slow to initialize (e.g., interactive prompts)
            except Exception as e:
                cprint(f"[AsyncSAM] Error during init_all_seg: {e}", "red")
                self.stop_all()
                raise

        cprint("[AsyncSAM] All SAM workers initialized segmentation.", "green")

    def submit_frames(self, rgb_frames: Dict[str, np.ndarray]):
        """
        Submits RGB frames to the worker processes for segmentation.
        This is non-blocking and respects the `sam_fps`.
        """
        if not self.started:
            cprint("[AsyncSAM] Workers not started, cannot submit frames.", "yellow")
            return

        current_time = time.perf_counter()
        if (current_time - self.last_submit_time) < (1.0 / self.sam_fps):
            return  # Don't submit too fast

        self.current_frame_id += 1
        for camera_name, rgb_frame in rgb_frames.items():
            try:
                self.input_queues[camera_name].put_nowait(
                    (self.current_frame_id, rgb_frame.copy())
                )
            except queue.Full:
                cprint(
                    f"[AsyncSAM] Input queue for {camera_name} is full, skipping frame {self.current_frame_id}",
                    "yellow",
                )
            except Exception as e:
                cprint(
                    f"[AsyncSAM] Error submitting frame for {camera_name}: {e}", "red"
                )

        self.last_submit_time = current_time

        # Continuously update latest masks from the output queue
        self._update_latest_masks()

    def get_latest_masks(self) -> Dict[str, Dict]:
        """
        Retrieves the latest available segmentation masks for all cameras.
        This is non-blocking.
        """
        self._update_latest_masks()  # Ensure latest masks are pulled from queue
        return self.latest_masks

    def _update_latest_masks(self):
        """
        Pulls all currently available masks from the output queue and updates
        the `self.latest_masks` dictionary.
        """
        while True:
            try:
                result = self.output_queue.get_nowait()
                cam_name = result["camera_name"]

                # Only update if the received mask is newer than the current one
                # or if the current one is uninitialized
                if result["frame_id"] > self.latest_masks[cam_name]["frame_id"]:
                    self.latest_masks[cam_name] = {
                        "binary_mask": result["binary_mask"],
                        "masks": result["masks"],
                        "age_ms": result["age_ms"],
                        "frame_id": result["frame_id"],
                    }
                # else:
                #     cprint(f"[AsyncSAM] Received older/same frame_id ({result['frame_id']}) for {cam_name}, current is {self.latest_masks[cam_name]['frame_id']}", "white")

            except queue.Empty:
                break  # No more items in the queue
            except Exception as e:
                cprint(f"[AsyncSAM] Error reading from output queue: {e}", "red")
                break


def create_async_sam_system(**kwargs) -> AsyncSamCameraManager:
    """
    Factory function to create an AsyncSamCameraManager.
    """
    return AsyncSamCameraManager(**kwargs)


if __name__ == "__main__":
    from contextlib import ExitStack, contextmanager
    from multiprocessing.managers import SharedMemoryManager

    import cv2

    from constants import CAMERA_CONFIGS
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
            manager = create_async_sam_system(
                cameras=active_cameras,
                camera_names=camera_names,
                sam_fps=5,
                model_type="b+",
                device_base="cuda",
            )

            camera_data = [None for _ in range(len(active_cameras))]
            for i in range(len(active_cameras)):
                camera_data[i] = active_cameras[i].get(out=camera_data[i])

            rgb_frames = {
                camera_names[i]: camera_data[i]["color"][..., ::-1]
                for i in range(len(camera_names))
            }

            # Initialize segmentation (blocking operation to ensure first masks are ready)
            manager.init_all_seg(rgb_frames)
            print("Initialized async SAM segmentation for all cameras")

            while True:
                time.sleep(0.1)  # simulate 10Hz control loop
                for i in range(len(active_cameras)):
                    camera_data[i] = active_cameras[i].get(out=camera_data[i])

                rgb_frames = {
                    camera_names[i]: camera_data[i]["color"][..., ::-1]
                    for i in range(len(camera_names))
                }

                manager.submit_frames(rgb_frames)
                latest_masks = manager.get_latest_masks()

                # concatenate binary masks horizontally
                binary_masks = [
                    latest_masks[camera_name]["binary_mask"].astype(np.float32)[
                        ..., None
                    ]
                    * 255
                    for camera_name in camera_names
                ]
                binary_masks = np.concatenate(binary_masks, axis=1)
                binary_masks = binary_masks.repeat(3, axis=-1).astype(np.uint8)

                rgb_frames = [
                    rgb_frames[camera_name][..., ::-1] for camera_name in camera_names
                ]
                rgb_frames = np.concatenate(rgb_frames, axis=1)

                combined_image = np.concatenate([rgb_frames, binary_masks], axis=0)
                cv2.imshow("combined_image", combined_image)
                cv2.waitKey(1)
