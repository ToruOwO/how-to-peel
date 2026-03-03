import enum
import math
import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager
from typing import List, Optional, Tuple, Callable, Generator

import av
import numpy as np

from shared_memory.shared_memory_queue import Empty, SharedMemoryQueue


def get_accumulate_timestamp_idxs(
    timestamps: List[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative=False,
) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def read_video(
    video_path: str,
    dt: float,
    video_start_time: float = 0.0,
    start_time: float = 0.0,
    img_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    thread_type: str = "AUTO",
    thread_count: int = 0,
    max_pad_frames: int = 10,
) -> Generator[np.ndarray, None, None]:
    frame = None
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        next_global_idx = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            # The presentation time in seconds for this frame.
            since_start = frame.time
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx,
            )
            if len(global_idxs) > 0:
                array = frame.to_ndarray(format="rgb24")
                img = array
                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield img
    # repeat last frame max_pad_frames times
    array = frame.to_ndarray(format="rgb24")
    img = array
    if img_transform is not None:
        img = img_transform(array)
    for i in range(max_pad_frames):
        yield img


class VideoRecorder(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    class Command(enum.Enum):
        START_RECORDING = 0
        STOP_RECORDING = 1

    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        buffer_size=128,
        no_repeat=False,
        # options for codec
        **kwargs,
    ):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.buffer_size = buffer_size
        self.no_repeat = no_repeat
        self.kwargs = kwargs

        self.img_queue = None
        self.cmd_queue = None
        self.stop_event = None
        self.ready_event = None
        self.is_started = False
        self.shape = None

        self._reset_state()

    # ======== custom constructors =======
    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile": profile},
            **kwargs,
        )
        return obj

    @classmethod
    def create_hevc_nvenc(
        cls,
        fps,
        codec="hevc_nvenc",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        bit_rate=6000 * 1000,
        options={"tune": "ll", "preset": "p1"},
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            bit_rate=bit_rate,
            options=options,
            **kwargs,
        )
        return obj

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, shm_manager: SharedMemoryManager, data_example: np.ndarray):
        super().__init__()
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.img_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={"img": data_example, "repeat": 1},
            buffer_size=self.buffer_size,
        )
        self.cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={
                "cmd": self.Command.START_RECORDING.value,
                "video_path": np.array("a" * self.MAX_PATH_LENGTH),
            },
            buffer_size=self.buffer_size,
        )
        self.shape = data_example.shape
        self.is_started = True
        super().start()

    def stop(self):
        self.stop_event.set()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    def is_ready(self):
        return (
            (self.start_time is not None)
            and (self.ready_event.is_set())
            and (not self.stop_event.is_set())
        )

    def start_recording(self, video_path: str, start_time: float = -1):
        path_len = len(video_path.encode("utf-8"))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError("video_path too long.")
        self.start_time = start_time
        self.cmd_queue.put(
            {"cmd": self.Command.START_RECORDING.value, "video_path": video_path}
        )

    def stop_recording(self):
        self.cmd_queue.put({"cmd": self.Command.STOP_RECORDING.value})
        self._reset_state()

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx = (
                get_accumulate_timestamp_idxs(
                    # only one timestamp
                    timestamps=[frame_time],
                    start_time=self.start_time,
                    dt=1 / self.fps,
                    next_global_idx=self.next_global_idx,
                )
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)

        self.img_queue.put({"img": img, "repeat": n_repeats})

    def get_img_buffer(self):
        """
        Get view to the next img queue memory
        for zero-copy writing
        """
        data = self.img_queue.get_next_view()
        img = data["img"]
        return img

    def write_img_buffer(self, img: np.ndarray, frame_time=None):
        """
        Must be used with the buffer returned by get_img_buffer
        for zero-copy writing
        """
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if (not self.no_repeat) and (self.start_time is not None):
            local_idxs, global_idxs, self.next_global_idx = (
                get_accumulate_timestamp_idxs(
                    # only one timestamp
                    timestamps=[frame_time],
                    start_time=self.start_time,
                    dt=1 / self.fps,
                    next_global_idx=self.next_global_idx,
                )
            )
            # number of apperance means repeats
            n_repeats = len(local_idxs)

        self.img_queue.put_next_view({"img": img, "repeat": n_repeats})

    # ========= interval API ===========
    def _reset_state(self):
        self.start_time = None
        self.next_global_idx = 0

    def run(self):
        # I'm sorry it has to be this complicated...
        self.ready_event.set()
        while not self.stop_event.is_set():
            video_path = None
            # ========= stopped state ============
            while (video_path is None) and (not self.stop_event.is_set()):
                try:
                    commands = self.cmd_queue.get_all()
                    for i in range(len(commands["cmd"])):
                        cmd = commands["cmd"][i]
                        if cmd == self.Command.START_RECORDING.value:
                            video_path = str(commands["video_path"][i])
                        elif cmd == self.Command.STOP_RECORDING.value:
                            video_path = None
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                except Empty:
                    time.sleep(0.1 / self.fps)
            if self.stop_event.is_set():
                break
            assert video_path is not None
            # ========= recording state ==========
            with av.open(video_path, mode="w") as container:
                stream = container.add_stream(self.codec, rate=self.fps)
                h, w, c = self.shape
                stream.width = w
                stream.height = h
                codec_context = stream.codec_context
                for k, v in self.kwargs.items():
                    setattr(codec_context, k, v)

                # loop
                while not self.stop_event.is_set():
                    try:
                        command = self.cmd_queue.get()
                        cmd = int(command["cmd"])
                        if cmd == self.Command.STOP_RECORDING.value:
                            break
                        elif cmd == self.Command.START_RECORDING.value:
                            continue
                        else:
                            raise RuntimeError("Unknown command: ", cmd)
                    except Empty:
                        pass

                    try:
                        with self.img_queue.get_view() as data:
                            img = data["img"]
                            repeat = data["repeat"]
                            frame = av.VideoFrame.from_ndarray(
                                img, format=self.input_pix_fmt
                            )
                        for _ in range(repeat):
                            for packet in stream.encode(frame):
                                container.mux(packet)
                    except Empty:
                        time.sleep(0.1 / self.fps)

                # Flush queue
                try:
                    while not self.img_queue.empty():
                        with self.img_queue.get_view() as data:
                            img = data["img"]
                            repeat = data["repeat"]
                            frame = av.VideoFrame.from_ndarray(
                                img, format=self.input_pix_fmt
                            )
                        for _ in range(repeat):
                            for packet in stream.encode(frame):
                                container.mux(packet)
                except Empty:
                    pass

                # Flush stream
                for packet in stream.encode():
                    container.mux(packet)
