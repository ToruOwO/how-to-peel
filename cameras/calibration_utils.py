import cv2
import numpy as np


def load_camera_info(serial_number, load_rgbd=True):
    info_dict = {}

    d = np.load(
        f"./data/camera_data/intrinsics_{serial_number}.npy",
        allow_pickle=True,
    ).item()
    fx = float(d["fx"])
    fy = float(d["fy"])
    ppx = float(d["ppx"])
    ppy = float(d["ppy"])
    intrinsic_matrix = np.array(
        [
            [fx, 0, ppx],
            [0, fy, ppy],
            [0, 0, 1.0],
        ],
        dtype=np.float32,
    )
    info_dict["intrinsics"] = d
    info_dict["intrinsic_matrix"] = intrinsic_matrix

    dist_coeffs = np.array(d["coeffs"], dtype=np.float32).reshape(-1, 1)
    width = int(d["width"])
    height = int(d["height"])
    extrinsics = np.load(
        f"./cameras/extrinsics/extrinsics_{serial_number}.npy"
    )
    info_dict["dist_coeffs"] = dist_coeffs
    info_dict["width"] = width
    info_dict["height"] = height
    info_dict["extrinsics"] = extrinsics

    if load_rgbd:
        depth = np.load(
            f"./data/camera_data/depth_{serial_number}.npy"
        )
        color = np.load(
            f"./data/camera_data/rgb_{serial_number}.npy"
        )
        info_dict["depth"] = depth
        info_dict["color"] = color
    return info_dict


def calib_downsample_points(
    points, if_crop=True, if_random_drop=True, num_points=300000
):
    # random drop points
    if if_random_drop:
        points = points[np.random.choice(points.shape[0], num_points, replace=False)]
    # visualizer.visualize_pointcloud(points)
    print(points.shape)
    # crop points
    if if_crop:
        points = points[
            np.where(
                (points[..., 0] > 0.05)
                & (points[..., 0] < 1)
                & (points[..., 1] > -1)
                & (points[..., 1] < 1)
                & (points[..., 2] > -0.05)
                & (points[..., 2] < 1.0)
            )
        ]
    print(points.shape)
    return points


def pose_delta(T_ref, T_new):
    R_ref, t_ref = T_ref[:3, :3], T_ref[:3, 3]
    R_new, t_new = T_new[:3, :3], T_new[:3, 3]

    R_delta = R_ref.T @ R_new
    t_delta = R_ref.T @ (t_new - t_ref)

    # rotation angle in radians from rotation matrix
    cos_theta = (np.trace(R_delta) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    return np.linalg.norm(t_delta), np.degrees(theta)


def aruco_check(rgb, camera_intrinsic, dist_coeffs, camera_extrinsics):
    aruco_pose_estimator = ArucoPoseEstimator(
        camera_intrinsic, dist_coeffs, aruco_dict_name=cv2.aruco.DICT_4X4_50
    )
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image, rvec, tvec = aruco_pose_estimator.visualize_pose(image, 0, 0.06)
    # T_world_tag:  Aug 25
    original_T_world_tag = np.array(
        [
            [0.03682479, -0.9993188, 0.00239333, 0.43896066],
            [0.99910758, 0.0368662, 0.02060869, -0.57551121],
            [-0.02068293, 0.00163233, 0.99978464, -0.01983361],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    if rvec is not None and tvec is not None:
        # Ensure correct shape for Rodrigues
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3] = tvec.ravel()
        T_world_tag = camera_extrinsics @ T_cam_tag
        print("T_world_tag: ", T_world_tag)

        # calculate deviation between original_T_world_tag and T_world_tag
        # Translation drift: ≤ 0.5–2.0 mm (0.0005–0.002 m)
        # Rotation drift: ≤ 0.03–0.2°
        # (rule of thumb: 0.1° at 1 m ≈ 1.7 mm lateral shift)
        trans_err_m, rot_err_deg = pose_delta(original_T_world_tag, T_world_tag)
        print("[ArucoCheck] trans_err_m: ", trans_err_m)
        print("[ArucoCheck] rot_err_deg: ", rot_err_deg)

        if trans_err_m > 0.002:
            print("[ArucoCheck] Translation drift is too large")
        if rot_err_deg > 0.2:
            print("[ArucoCheck] Rotation drift is too large")
    else:
        print("[ArucoCheck] did not find the target")


class ArucoPoseEstimator:
    def __init__(
        self, camera_intrinsic, dist_coeffs, aruco_dict_name=cv2.aruco.DICT_4X4_50
    ):
        self.camera_intrinsic = camera_intrinsic
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, cv2.aruco.DetectorParameters()
        )

    def _estimate_pose(self, image, target_id, marker_length):
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or target_id not in ids:
            return None, None, None, None
        index = np.where(ids == target_id)[0][0]

        # Define the 3D coordinates of the marker corners in its own coordinate system
        obj_points = np.array(
            [
                [-marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0],
            ],
            dtype=np.float32,
        )

        # 2D image points from detection
        img_points = corners[index].reshape(-1, 2).astype(np.float32)

        success, rvecs, tvecs = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_intrinsic,
            self.dist_coeffs,
        )

        if not success:
            print("[WARN] solvePnP failed")
            return None, None, None, None
        return rvecs, tvecs, corners, ids

    def _rvec_tvec_to_matrix(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        return T

    def estimate_pose_cam_frame(self, image, target_id, marker_length):
        try:
            rvec, tvec, _, _ = self._estimate_pose(image, target_id, marker_length)
            return self._rvec_tvec_to_matrix(rvec, tvec)
        except Exception as e:
            print(f"[ArucoPoseEstimator] Error estimating pose: {e}")
            return None

    def visualize_pose(self, image, target_id, marker_length):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        def draw_custom_axis(
            img, rvec, tvec, camera_intrinsic, dist_coeffs, axis_length
        ):
            axis_3d = np.float32(
                [
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length],
                ]
            )
            imgpts, _ = cv2.projectPoints(
                axis_3d, rvec, tvec, camera_intrinsic, dist_coeffs
            )
            imgpts = imgpts.astype(int).reshape(-1, 2)

            origin = tuple(imgpts[0])
            img = cv2.line(img, origin, tuple(imgpts[1]), (0, 0, 255), 2)
            img = cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 2)
            img = cv2.line(img, origin, tuple(imgpts[3]), (255, 0, 0), 2)
            return img

        rvec, tvec, corners, ids = self._estimate_pose(image, target_id, marker_length)
        if corners is not None and ids is not None:
            image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # Guard when target not found
        if rvec is None or tvec is None:
            print(
                f"[ArucoPoseEstimator] Target ID {target_id} not found. Skipping pose drawing."
            )
            return image, None, None
        # Normalize shapes for OpenCV
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        image = draw_custom_axis(
            image,
            rvec,
            tvec,
            self.camera_intrinsic,
            self.dist_coeffs,
            marker_length * 0.5,
        )
        return image, rvec, tvec
