import enum
import fcntl
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from termcolor import cprint
from threadpoolctl import threadpool_limits

from shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
    SharedMemoryRingBufferView,
)
from shared_memory.shared_ndarray import SharedNDArray


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number,
        product_line,
        depth_resolution=(640, 480),
        color_resolution=(640, 480),
        capture_fps=30,
        put_fps=None,
        enable_color=True,
        enable_depth=False,
        get_max_k=30,
        transform: Optional[Callable[[Dict], Dict]] = None,
        verbose=False,
    ):
        super().__init__()
        self.shm_manager = shm_manager

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        # NOTE: all frames are aligned to color resolution
        color_resolution = tuple(color_resolution)
        color_shape = color_resolution[::-1]
        examples = dict()
        if enable_color:
            examples["color"] = np.empty(shape=color_shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=color_shape, dtype=np.uint16)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(12,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0

        # copied variables
        self.serial_number = serial_number
        self.product_line = product_line
        self.depth_resolution = depth_resolution
        self.color_resolution = color_resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.transform = transform
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices_info():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                serials.append((serial, product_line))
        serials = sorted(serials, key=lambda x: x[0])
        return serials

    @staticmethod
    def reset_device(serial_number):
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.serial_number) == serial_number:
                d.hardware_reset()
                cprint(
                    f"[RealSense] Reset {d.get_info(rs.camera_info.serial_number)}",
                    "green",
                )
                break
        else:
            cprint(
                f"[RealSense] No RealSense device found with serial number {serial_number}",
                "red",
            )

    @staticmethod
    def reset_usb_port(product_model):
        try:
            output = subprocess.check_output("lsusb").decode()
        except subprocess.CalledProcessError as e:
            cprint(f"[RealSense] lsusb failed: {e}", "red")
            return

        found = False
        for line in output.strip().split("\n"):
            if (
                "Intel Corp." in line
                and "RealSense(TM)" in line
                and product_model in line
            ):
                match = re.search(r"Bus (\d{3}) Device (\d{3}):", line)
                if match:
                    found = True
                    bus = match.group(1)
                    device = match.group(2)
                    dev_path = f"/dev/bus/usb/{bus}/{device}"
                    print("[RealSense] USB Device Path: ", dev_path)
                    USBDEVFS_RESET = 21780
                    try:
                        with open(dev_path, "wb") as f:
                            fcntl.ioctl(f, USBDEVFS_RESET, 0)
                        cprint(f"[RealSense] Successfully reset {dev_path}", "green")
                    except Exception as e:
                        cprint(f"[RealSense] Failed to reset {dev_path}: {e}", "red")
        if not found:
            cprint("[RealSense] No RealSense device found", "red")
        exit()

    # ========= context manager ===========

    @contextmanager
    def activate(self):
        started = False
        try:
            self.start()
            started = True
            yield self
        finally:
            if started:
                try:
                    self.stop()
                except Exception:
                    pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def save_proc_specs(self, file_path="./cameras/configs"):
        proc_specs = dict()
        proc_specs["version"] = 1  # schema version for forward compatibility

        # instance specs
        proc_specs["serial_number"] = self.serial_number
        proc_specs["put_fps"] = self.put_fps
        proc_specs["get_max_k"] = self.ring_buffer.get_max_k

        # shared memory specs
        proc_specs["ring_buffer"] = dict()
        for key in self.ring_buffer.shared_arrays.keys():
            proc_specs["ring_buffer"][key] = {
                "proc_name": self.ring_buffer.shared_arrays[key].shm.name,
                "shape": list(self.ring_buffer.shared_arrays[key].shape),
                "dtype": str(self.ring_buffer.shared_arrays[key].dtype),
            }
        proc_specs["ring_buffer"]["counter"] = {
            "proc_name": self.ring_buffer.counter.shm.name,
            "size": self.ring_buffer.counter.size,
        }

        # save process name, shape, dtype to json
        with open(
            os.path.join(file_path, f"proc_specs_{self.serial_number}.json"), "w"
        ) as f:
            json.dump(proc_specs, f)
        print(
            f"[CameraServer] Saved proc specs to {os.path.join(file_path, f'proc_specs_{self.serial_number}.json')}"
        )

    @classmethod
    def attach_from_proc_specs(  # WIP
        cls,
        serial_number,
        specs_file_path="./cameras/configs",
    ):
        with open(
            os.path.join(specs_file_path, f"proc_specs_{serial_number}.json"), "r"
        ) as f:
            proc_specs = json.load(f)

        assert proc_specs["version"] == 1
        assert proc_specs["serial_number"] == serial_number

        put_fps = proc_specs["put_fps"]
        get_max_k = proc_specs["get_max_k"]

        # Create a placeholder instance without allocating new shared arrays.
        # use __new__ to skip __init__ (which allocates).
        instance = object.__new__(cls)

        # recover ring buffer
        ring_buffer = SharedMemoryRingBufferView(
            proc_specs=proc_specs["ring_buffer"],
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )
        instance.ring_buffer = ring_buffer

        return instance

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time

        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        # only join if we were started
        try:
            if getattr(self, "_popen", None) is not None:
                super().join(timeout=2)
        except Exception:
            pass

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put(
            {
                "cmd": Command.SET_COLOR_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_intrinsics_dict(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        dist_coeffs = self.intrinsics_array.get()[6:11]
        scale = self.intrinsics_array.get()[11]
        return {
            "coeffs": dist_coeffs,
            "fx": fx,
            "fy": fy,
            "height": self.color_resolution[1],
            "model": "inverse_brown_conrady",
            "ppx": ppx,
            "ppy": ppy,
            "width": self.color_resolution[0],
            "scale": scale,
        }

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[11]
        return scale

    def start_recording(self, video_path: str, start_time: float = -1):
        assert self.enable_color

        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.command_queue.put(
            {
                "cmd": Command.START_RECORDING.value,
                "video_path": video_path,
                "recording_start_time": start_time,
            }
        )

    def stop_recording(self):
        self.command_queue.put({"cmd": Command.STOP_RECORDING.value})

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    # ========= interval API ===========

    def get_depth_frame(self, frameset):
        depth_frame = frameset.get_depth_frame()
        depth = np.asarray(depth_frame.get_data())
        return depth

    def get_color_frame(self, frameset):
        color_frame = frameset.get_color_frame()
        color = np.asarray(color_frame.get_data())
        return color

    def warmup(self, pipeline, align):
        if self.product_line == "L500":
            prepare_time, warmup_time = 10, 100
        else:
            prepare_time, warmup_time = 1, 10

        for _ in range(prepare_time):
            try:
                _ = pipeline.wait_for_frames()
            except Exception as e:
                cprint(f"[RealSense] Prepare Error:{e}", "red")
                cprint(
                    f"[RealSense] {self.serial_number} Start Resetting Hardware", "red"
                )
                self.reset_device(self.serial_number)
                time.sleep(1)

        for _ in range(warmup_time):
            try:
                frames = pipeline.wait_for_frames()
                if not frames:
                    cprint(f"[RealSense] {self.serial_number} No frames", "red")

                if self.enable_depth:
                    aligned_frames = align.process(frames)
                    _ = self.get_depth_frame(aligned_frames)
                    _ = self.get_color_frame(aligned_frames)
                else:
                    _ = self.get_color_frame(frames)
            except Exception as e:
                cprint(f"[RealSense] Warmup Error:{e}", "red")
                cprint(
                    f"[RealSense] {self.serial_number} Start Resetting Hardware", "red"
                )
                cprint("[RealSense] Start Resetting Hardware", "red")
                self.reset_device(self.serial_number)
                time.sleep(0.01)

        print(f"[RealSense] {self.serial_number} Warmup Done")

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        color_w, color_h = self.color_resolution
        depth_w, depth_h = self.depth_resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)

        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(
                rs.stream.color, color_w, color_h, rs.format.bgr8, fps
            )
        if self.enable_depth:
            rs_config.enable_stream(
                rs.stream.depth, depth_w, depth_h, rs.format.z16, fps
            )

        try:
            rs_config.enable_device(str(self.serial_number))

            # start pipeline
            pipeline = rs.pipeline()

            # NOTE: this is a hack for L515, which sometimes fails to start the pipeline.
            try:
                pipeline_profile = pipeline.start(rs_config)
            except Exception as e:
                cprint(
                    f"[RealSense] {self.product_line} Failed to start pipeline: {e}",
                    "red",
                )
                self.reset_usb_port("405" if self.product_line == "D400" else "515")

            # get intrinsics
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ["fx", "fy", "ppx", "ppy", "height", "width"]
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)
            self.intrinsics_array.get()[6:11] = np.array(intr.coeffs)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[11] = depth_scale

            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

            # make sure frames are ready
            self.warmup(pipeline, align)

            iter_idx = 0
            t_start = time.time()
            t10 = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = pipeline.wait_for_frames()

                frameset = align.process(frameset)

                # grab data
                data = dict()

                if self.enable_color:
                    data["color"] = self.get_color_frame(frameset)
                if self.enable_depth:
                    data["depth"] = self.get_depth_frame(frameset)

                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                try:
                    # NOTE: L500 is weird
                    self.ring_buffer.put(put_data, wait=(self.product_line == "L500"))
                except Exception as e:
                    cprint(f"[SingleRealsense] Error putting data: {e}", "red")
                    for k, v in put_data.items():
                        print(f"{k}: {v.shape}")
                    raise

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")

                iter_idx += 1
                if self.verbose:
                    if iter_idx == 10:
                        t10 = time.time()
                    elif iter_idx > 10:
                        print((iter_idx - 10) / (time.time() - t10))
        finally:
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")
