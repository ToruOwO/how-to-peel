import numpy as np

# RPC servers
ARM_RPC_HOST = "192.168.1.21"
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b"secret password"

# Env client
ARM_CONTROL_FREQ = 1000
ARM_CONTROL_PERIOD = 1.0 / ARM_CONTROL_FREQ

CONTROLLER_CONFIGS = {
    "original": {
        "ALPHA": 0.01,
        "K_r": np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18]),
        "K_l": np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0]),
        "K_lp": np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0]),
        "K_p": np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        "K_d": np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0]),
    },
    "ours0": {
        "ALPHA": 0.01,
        "K_r": np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18]),
        "K_l": np.diag([106.2, 100.8, 106.2, 106.2, 131.4, 106.2, 106.2]),
        "K_lp": np.diag([11.89, 25.52, 22.0, 22.0, 22.0, 22.0, 22.0]),
        "K_p": np.diag([382.2, 296.4, 347.1, 400.0, 200.0, 200.0, 200.0]),
        "K_d": np.diag([21.0, 17.5, 10.0, 10.0, 5.0, 5.0, 5.0]),
    },
    "pos": {
        "ALPHA": 0.01,
        "K_r": np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18]),
        "K_l": np.diag([10.0, 10.0, 10.0, 10.0, 12.0, 10.0, 10.0]),
        "K_lp": np.diag([1.0, 2.5, 2.2, 2.2, 2.2, 2.2, 2.2]),
        "K_p": np.diag([600.0, 500.0, 550.0, 600.0, 300.0, 300.0, 300.0]),
        "K_d": np.diag([40.0, 30.0, 20.0, 20.0, 10.0, 10.0, 10.0]),
    },
}
DEFAULT_CONTROLLER_CONFIG = "ours0"

# Cameras
CAMERA_CONFIGS = {
    "lidar": {
        "serial_number": "f1422047",
        "product_line": "L500",
        "depth_resolution": (1024, 768),
        "color_resolution": (1920, 1080),
        "capture_fps": 30,
        "enable_color": True,
        "enable_depth": True,
    },
    "wrist": {
        "serial_number": "218622270598",
        "product_line": "D400",
        "depth_resolution": (640, 480),
        "color_resolution": (640, 480),
        "capture_fps": 30,
        "enable_color": True,
        "enable_depth": True,
    },
    "wrist2": {
        "serial_number": "123622270575",
        "product_line": "D400",
        "depth_resolution": (640, 480),
        "color_resolution": (640, 480),
        "capture_fps": 30,
        "enable_color": True,
        "enable_depth": True,
    },
}

LIDAR_ASSET_PATHS = [
    "./data/camera_data/rgb_f1422047.npy",
    "./data/camera_data/rgb_f1422047.png",
    "./data/camera_data/depth_f1422047.npy",
    "./data/camera_data/depth_f1422047.png",
]

# Policy
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

DATASET_VERSION = 3  # 2: single camera; 1: mask; 0: initial

RESET_CMD_DEG = {
    "long": np.array([29.54, 13.22, 340.33, 116.07, 353.83, 51.27, 276.96]),
    "short": np.array([25.31, 13.51, 344.67, 111.96, 13.96, 53.35, 264.7]),
}

RESET_CMD = {
    "long": np.deg2rad(RESET_CMD_DEG["long"]),
    "short": np.deg2rad(RESET_CMD_DEG["short"]),
}
