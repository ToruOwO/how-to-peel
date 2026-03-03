#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from termcolor import cprint

from cameras.pc_visualizer import Visualizer


def get_rotation_matrix_from_two_vectors(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    if c == -1:
        # Opposite direction
        return -np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2 + 1e-10))
    return R


def transform_pointcloud(pointcloud, rotation_matrix):
    point_xyz = pointcloud[:, :3]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = rotation_matrix @ point_homogeneous.T
    point_xyz = point_homogeneous[:3, :].T
    pointcloud = np.hstack((point_xyz, pointcloud[:, 3:]))
    return pointcloud


def create_point_cloud(
    depth,
    color,
    intrinsics,
    extrinsics_matrix,
    segment_mask=None,
    apply_dbscan=True,
    dbscan_eps=0.006,
    dbscan_min_samples=200,
    debug=False,
):
    """
    Create point cloud from depth and color.

    Args:
        depth (np.ndarray): Depth image, (H, W).
        color (np.ndarray): Color image, (H, W, 3).
        intrinsics (dict): Intrinsics dictionary.
                            with keys: "fx", "fy", "ppx", "ppy", "width", "height", "scale"
        extrinsics_matrix (np.ndarray): 4x4 extrinsics matrix.
        segment_mask (np.ndarray): Segmentation mask, (H, W).
        apply_dbscan (bool): Whether to apply DBSCAN clustering.
        dbscan_eps (float): DBSCAN epsilon.
        dbscan_min_samples (int): DBSCAN min samples.
        debug (bool): Whether to print debug information.

    Returns:
        np.ndarray: Point cloud, (N, 6), with XYZRGB.
    """
    assert depth.shape[:2] == (intrinsics["height"], intrinsics["width"]), (
        "Depth Shape: {}, Intrinsics Shape: {}".format(
            depth.shape[:2], (intrinsics["height"], intrinsics["width"])
        )
    )

    # Segmentation mask
    if segment_mask is not None:
        mask = segment_mask
        color = color * mask[..., None]
        depth = depth * mask
    else:
        mask = np.ones(depth.shape, dtype=bool)

    start_time = time.time()

    # Point cloud generation
    xmap = np.arange(intrinsics["width"])
    ymap = np.arange(intrinsics["height"])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth * intrinsics["scale"]
    points_x = (xmap - intrinsics["ppx"]) * points_z / intrinsics["fx"]
    points_y = (ymap - intrinsics["ppy"]) * points_z / intrinsics["fy"]

    point_xyz = np.stack([points_x, points_y, points_z], axis=-1)
    point_color = color

    if debug:
        # visualize segmented color and depth
        point_z_vis = cv.applyColorMap(
            cv.convertScaleAbs(points_z, alpha=0.03), cv.COLORMAP_JET
        )
        color_vis = np.concatenate([color, point_z_vis], axis=1)
        cv.imshow("vis", color_vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Flatten
    point_xyz = point_xyz.reshape(-1, 3)
    point_color = point_color.reshape(-1, 3)

    if debug:
        # visualize again
        point_z_vis = cv.applyColorMap(
            cv.convertScaleAbs(
                point_xyz.reshape(intrinsics["height"], intrinsics["width"], 3)[
                    :, :, 2
                ],
                alpha=0.03,
            ),
            cv.COLORMAP_JET,
        )
        color_vis = np.concatenate(
            [
                point_color.reshape(intrinsics["height"], intrinsics["width"], 3),
                point_z_vis,
            ],
            axis=1,
        )
        cv.imshow("vis", color_vis)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Remove invalid (depth=0 or nan)
    valid = (point_xyz[:, 2] > 0) & (~np.isnan(point_xyz).any(axis=1))
    point_xyz = point_xyz[valid]
    point_color = point_color[valid]

    if debug:
        viz = Visualizer()
        viz.visualize_pointcloud(point_xyz)
        time.sleep(10)

    # DBSCAN filtering
    if apply_dbscan:
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(
            point_xyz
        )
        keep = labels != -1
        point_xyz = point_xyz[keep]
        point_color = point_color[keep]
        if debug:
            viz = Visualizer()
            viz.visualize_pointcloud(point_xyz)

    pointcloud = np.hstack([point_xyz, point_color])
    pointcloud = transform_pointcloud(pointcloud, extrinsics_matrix)

    if debug:
        end_time = time.time()
        print(f"[RealSense] Time taken: {end_time - start_time} seconds")
        print(f"[RealSense] Pointcloud size: {pointcloud.shape}")

    return pointcloud


def o3d_pcd_with_normals(
    pointcloud,
    normal_radius=0.01,
    normal_max_nn=2000,
    camera_location=[0.9, -0.25, 0.6],
):
    """
    Estimate normals and orient them consistently.

    Args:
        pointcloud (np.ndarray): Nx3 or Nx6, with XYZ or XYZRGB.
        normal_radius (float): Radius used for normal estimation.
        normal_max_nn (int): Max neighbors for normal estimation.
        camera_location (np.ndarray): Camera location for normal orientation.

    Returns:
        o3d.geometry.PointCloud: Point cloud with estimated + oriented normals.
    """
    xyz = pointcloud[:, :3]
    rgb = pointcloud[:, 3:] / 255.0 if pointcloud.shape[1] > 3 else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    camera_location = (
        np.array(camera_location)
        if isinstance(camera_location, list)
        else camera_location
    )
    pcd.orient_normals_towards_camera_location(camera_location=camera_location)

    return pcd


def normal_to_rotation_matrix(normal: np.ndarray) -> np.ndarray:
    """
    Given a normal vector, construct a rotation matrix where z axis aligns with the normal.
    """
    z = normal / np.linalg.norm(normal)
    # Choose x to be orthogonal to z
    x = np.array([1, 0, 0])
    if np.abs(np.dot(x, z)) > 0.9:
        x = np.array([0, 1, 0])
    x = x - np.dot(x, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    return R


def visualize_trajectory_on_pcd(
    pcd_o3d,
    traj_points,
    normals=None,
    show_axes=False,
    axes_size=0.05,
    axes_origin="origin",
):
    """
    Visualize point cloud with trajectory points and normals.

    Args:
        pcd_o3d: open3d.geometry.PointCloud
        traj_points: list or np.ndarray of (N,3) points
        normals: (optional) list or np.ndarray of (N,3) normals
        show_axes: whether to display XYZ coordinate frame
        axes_size: size of the coordinate frame
        axes_origin: one of {"origin", "centroid", "first"}
        "origin": origin at (0,0,0)
        "centroid": origin at the centroid of the trajectory points
        "first": origin at the first point of the trajectory
    """
    vis_list = [pcd_o3d]

    # Optional XYZ axes
    if show_axes:
        if axes_origin == "centroid" and len(traj_points) > 0:
            origin = np.mean(np.asarray(traj_points), axis=0)
        elif axes_origin == "first" and len(traj_points) > 0:
            origin = np.asarray(traj_points)[0]
        else:
            origin = np.zeros(3)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size)
        axes.translate(origin)
        vis_list.append(axes)

    # Trajectory as red spheres
    for pt in traj_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(pt)
        vis_list.append(sphere)

    # Normals as arrows
    if normals is not None:
        for pt, n in zip(traj_points, normals):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.001,
                cone_radius=0.002,
                cylinder_height=0.01,
                cone_height=0.005,
            )
            Rm = normal_to_rotation_matrix(n)
            arrow.rotate(Rm, center=np.array([0, 0, 0]))
            arrow.translate(pt)
            arrow.paint_uniform_color([0, 1, 0])
            vis_list.append(arrow)

    o3d.visualization.draw_geometries(vis_list)


class TrajectoryProcessor:
    def __init__(
        self,
        pcd_o3d,
        traj_reserve_ratio=0.8,
        fit_x_window=None,
        fit_zmax_ratio=None,
        curve_num_samples=500,
        curve_num_out=200,
        visualize=True,
    ):
        self.pcd_o3d = pcd_o3d
        self.tree = o3d.geometry.KDTreeFlann(self.pcd_o3d)
        self.num_samples = curve_num_samples
        self.num_out = curve_num_out
        self.traj_reserve_ratio = traj_reserve_ratio

        # fit the parabola
        self.fit_x_window = fit_x_window  # using points in x-range to fit the parabola
        self.fit_zmax_ratio = fit_zmax_ratio  # using the highest z values in the point cloud to fit the parabola
        self.ref_traj = self.generate_ref_traj(self.traj_reserve_ratio)
        if visualize:
            visualize_trajectory_on_pcd(
                self.pcd_o3d,
                self.ref_traj[:, :3],
                self.ref_traj[:, 3:],
                show_axes=True,
                axes_size=0.05,
                axes_origin="centroid",
            )

    def _uniform_sample_along_curve(self, proj_sorted, num_samples):
        diffs = np.diff(proj_sorted, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        cumdist = np.insert(np.cumsum(dists), 0, 0.0)
        if cumdist[-1] <= 1e-9:
            raise RuntimeError(
                "Degenerate point cloud projection: zero length along path"
            )
        target = np.linspace(0.0, cumdist[-1], num_samples)
        x_interp = np.interp(target, cumdist, proj_sorted[:, 0])
        y_interp = np.interp(target, cumdist, proj_sorted[:, 1])
        z_interp = np.interp(target, cumdist, proj_sorted[:, 2])
        return x_interp, y_interp, z_interp, cumdist

    def _parabola_fit_curve_along_y_minus(self, xi, yi, zi, n_out):
        coeff_x = np.polyfit(yi, xi, 2)
        coeff_z = np.polyfit(yi, zi, 2)
        y_fine = np.linspace(yi.max(), yi.min(), n_out)
        x_fine = np.polyval(coeff_x, y_fine)
        z_fine = np.polyval(coeff_z, y_fine)
        return np.vstack([x_fine, y_fine, z_fine]).T

    def _project_curve_onto_surface(
        self, points: np.ndarray, curve_pts: np.ndarray
    ) -> np.ndarray:
        if curve_pts.shape[0] == 0:
            return curve_pts.copy()
        if points.shape[0] == 0:
            return curve_pts.copy()

        num_curve_points = curve_pts.shape[0]
        num_cloud_points = points.shape[0]

        if num_cloud_points < 3:
            snap_idx = []
            for i in range(curve_pts.shape[0]):
                _, idx, _ = self.tree.search_knn_vector_3d(curve_pts[i], 1)
                snap_idx.append(idx[0])
            return points[np.asarray(snap_idx, dtype=int)]

        k_neighbors = min(30, max(5, num_cloud_points // 100))
        if k_neighbors < 3:
            snap_idx = []
            for i in range(curve_pts.shape[0]):
                _, idx, _ = self.tree.search_knn_vector_3d(curve_pts[i], 1)
                snap_idx.append(idx[0])
            return points[np.asarray(snap_idx, dtype=int)]

        projected_points = np.empty_like(curve_pts)

        for i in range(num_curve_points):
            query_point = curve_pts[i]

            k, idx, dists2 = self.tree.search_knn_vector_3d(query_point, k_neighbors)
            dists = np.sqrt(np.asarray(dists2))
            neighbor_points = points[np.asarray(idx, dtype=int)]
            if neighbor_points.ndim == 1:
                projected_points[i] = neighbor_points
                continue

            sigma = max(1e-6, np.median(dists))
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            weights = np.clip(weights, 1e-6, None)
            weights = weights / np.sum(weights)

            weighted_centroid = np.sum(neighbor_points * weights[:, None], axis=0)

            centered = neighbor_points - weighted_centroid
            weighted_centered = centered * np.sqrt(weights[:, None])
            try:
                _, _, vt = np.linalg.svd(weighted_centered, full_matrices=False)
                normal_vector = vt[-1]
            except np.linalg.LinAlgError:
                projected_points[i] = weighted_centroid
                continue

            diff = query_point - weighted_centroid
            offset = np.dot(diff, normal_vector) * normal_vector
            plane_projection = query_point - offset

            blend_ratio = 0.2
            projected_points[i] = (
                1.0 - blend_ratio
            ) * plane_projection + blend_ratio * weighted_centroid

        n = projected_points.shape[0]
        if n >= 7:
            window_length = min(21, n if (n % 2 == 1) else n - 1)
            window_length = max(7, window_length)
            poly_order = 2 if window_length >= 5 else 1
            for dim in range(3):
                projected_points[:, dim] = savgol_filter(
                    projected_points[:, dim],
                    window_length=window_length,
                    polyorder=poly_order,
                    mode="interp",
                )

        snap_idx = []
        for i in range(projected_points.shape[0]):
            _, idx, _ = self.tree.search_knn_vector_3d(projected_points[i], 1)
            snap_idx.append(idx[0])
        snapped_points = points[np.asarray(snap_idx, dtype=int)]

        return snapped_points

    def _smooth_trajectory(self, traj, window=11, poly=3):
        traj_smooth = np.zeros_like(traj)
        for i in range(3):
            traj_smooth[:, i] = savgol_filter(
                traj[:, i], window_length=window, polyorder=poly
            )
        return traj_smooth

    def generate_ref_traj(self, ratio=None, visualize=False):
        points_xyz = np.asarray(self.pcd_o3d.points)
        order = np.argsort(points_xyz[:, 0])
        points_sorted = points_xyz[order]
        xi_sel, yi_sel, zi_sel = (
            points_sorted[:, 0],
            points_sorted[:, 1],
            points_sorted[:, 2],
        )

        if self.fit_x_window is not None:
            # Calculate xmin and xmax based on ratio
            x_range = xi_sel.max() - xi_sel.min()
            xmin = xi_sel.min() + self.fit_x_window[0] * x_range
            xmax = xi_sel.min() + self.fit_x_window[1] * x_range
            mask = (xi_sel >= xmin) & (xi_sel <= xmax)
            if np.count_nonzero(mask) >= 3:
                xi_sel, yi_sel, zi_sel = xi_sel[mask], yi_sel[mask], zi_sel[mask]

                # Visualize selected (red) and unselected (gray) points
                if visualize:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection="3d")

                    # Plot unselected points in gray
                    unselected_mask = ~mask
                    ax.scatter(
                        points_sorted[unselected_mask, 0],
                        points_sorted[unselected_mask, 1],
                        points_sorted[unselected_mask, 2],
                        c="gray",
                        label="Unselected points",
                        alpha=0.5,
                        s=1,
                    )

                    # Plot selected points in red
                    ax.scatter(
                        points_sorted[mask, 0],
                        points_sorted[mask, 1],
                        points_sorted[mask, 2],
                        c="red",
                        label="Selected points",
                        alpha=1.0,
                        s=1,
                    )

                    ax.set_xlabel("X (real size)")
                    ax.set_ylabel("Y (real size)")
                    ax.set_zlabel("Z (real size)")
                    ax.set_title("Selected vs. Unselected Points (fit_x_window)")
                    ax.legend()
                    plt.show()

        curve = self._parabola_fit_curve_along_y_minus(
            xi_sel, yi_sel, zi_sel, n_out=self.num_out
        )
        ref_traj_points = self._project_curve_onto_surface(points_xyz, curve)

        # fetch normals by nearest neighbor index for each trajectory point
        ref_traj_normals_list = []
        ref_traj_points_exact_list = []  # points on the curve
        normals_np = np.asarray(self.pcd_o3d.normals)
        for p in ref_traj_points:
            _, idx, _ = self.tree.search_knn_vector_3d(p, 1)
            ref_traj_normals_list.append(-normals_np[idx[0]])
            ref_traj_points_exact_list.append(points_xyz[idx[0]])

        ref_traj_normals = np.asarray(ref_traj_normals_list)
        ref_traj_points_exact = np.asarray(ref_traj_points_exact_list)

        # save the ratio in the middle of the trajectory
        if ratio is None:
            ratio = self.traj_reserve_ratio
        ref_traj_points_exact = ref_traj_points_exact[
            int(len(ref_traj_points_exact) * (1 - ratio) / 2) : int(
                len(ref_traj_points_exact) * (1 + ratio) / 2
            )
        ]
        ref_traj_normals = ref_traj_normals[
            int(len(ref_traj_normals) * (1 - ratio) / 2) : int(
                len(ref_traj_normals) * (1 + ratio) / 2
            )
        ]

        ref_traj = np.concatenate([ref_traj_points_exact, ref_traj_normals], axis=1)
        return ref_traj

    def get_normal(self, point):
        [k, idx, _] = self.tree.search_knn_vector_3d(point, 1)
        #! make the normal pointing to the inside of the object
        # calculated normal points down but we want knife normal to point up, thus the minus sign
        return -np.asarray(self.pcd_o3d.normals)[idx[0]]

    def get_tangent(
        self, point, method="weighted_average", smooth_window=11, smooth_poly=3
    ):
        traj_smooth = self._smooth_trajectory(
            self.ref_traj[:, :3], smooth_window, smooth_poly
        )
        dists = np.linalg.norm(traj_smooth - point, axis=1)
        closest_idx = np.argmin(dists)

        if method == "local_average":
            window_size = min(20, len(traj_smooth) // 4)
            start_idx = max(0, closest_idx - window_size // 2)
            end_idx = min(len(traj_smooth), closest_idx + window_size // 2 + 1)

            tangent_vectors = []
            for i in range(start_idx, end_idx - 1):
                if i + 1 < len(traj_smooth):
                    local_tangent = traj_smooth[i + 1] - traj_smooth[i]
                    if np.linalg.norm(local_tangent) > 1e-6:
                        tangent_vectors.append(
                            local_tangent / np.linalg.norm(local_tangent)
                        )
            if tangent_vectors:
                tangent_vector = np.mean(tangent_vectors, axis=0)
            else:
                tangent_vector = np.array([1.0, 0.0, 0.0])

        elif method == "weighted_average":
            window_size = min(25, len(traj_smooth) // 3)
            start_idx = max(0, closest_idx - window_size // 2)
            end_idx = min(len(traj_smooth), closest_idx + window_size // 2 + 1)

            tangent_vectors = []
            weights = []

            for i in range(start_idx, end_idx - 1):
                if i + 1 < len(traj_smooth):
                    local_tangent = traj_smooth[i + 1] - traj_smooth[i]
                    if np.linalg.norm(local_tangent) > 1e-6:
                        mid_point = (traj_smooth[i] + traj_smooth[i + 1]) / 2
                        dist_to_query = np.linalg.norm(mid_point - point)

                        sigma = 0.02
                        weight = np.exp(-(dist_to_query**2) / (2 * sigma**2))

                        tangent_vectors.append(
                            local_tangent / np.linalg.norm(local_tangent)
                        )
                        weights.append(weight)

            if tangent_vectors and weights:
                weights = np.array(weights)
                weights /= np.sum(weights)
                tangent_vector = np.average(tangent_vectors, axis=0, weights=weights)
            else:
                tangent_vector = np.array([1.0, 0.0, 0.0])
        else:
            raise ValueError(f"Invalid method: {method}")

        tangent_norm = np.linalg.norm(tangent_vector)
        if tangent_norm > 1e-6:
            return tangent_vector / tangent_norm
        else:
            return np.array([1.0, 0.0, 0.0])

    def get_closest_ref_traj_point(self, point):
        dists = np.linalg.norm(self.ref_traj[:, :3] - point, axis=1)
        closest_idx = np.argmin(dists)
        return self.ref_traj[closest_idx, :3]

    def visualize_ref_traj_tangents(
        self, num_samples: int = 10, arrow_length: float = 0.01
    ):
        if self.ref_traj is None or len(self.ref_traj) == 0:
            cprint("[Warn] ref_traj is empty. Nothing to visualize.", "yellow")
            return
        num_samples = max(1, int(num_samples))
        indices = np.linspace(0, len(self.ref_traj) - 1, num_samples, dtype=int)

        geometries = []
        geometries.append(self.pcd_o3d)

        for idx in indices:
            point = self.ref_traj[idx, :3]
            tangent = self.get_tangent(point)
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-9:
                continue
            tangent = tangent / tangent_norm

            # sample point sphere
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            sp.translate(point)
            sp.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
            geometries.append(sp)

            # tangent arrow (Open3D arrow is along +Z by default)
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.0005,
                cone_radius=0.001,
                cylinder_height=arrow_length * 0.8,
                cone_height=arrow_length * 0.2,
            )
            arrow.paint_uniform_color([1.0, 0.65, 0.0])  # orange
            arrow.translate(point)
            R = get_rotation_matrix_from_two_vectors(np.array([0.0, 0.0, 1.0]), tangent)
            arrow.rotate(R, center=point)
            geometries.append(arrow)

        o3d.visualization.draw_geometries(geometries)

    def get_start_point(self, delta_to_normal=75):
        start_xyz = self.ref_traj[0, :3]

        normal = self.get_normal(start_xyz)
        normal = normal / np.linalg.norm(normal)

        tangent = self.get_tangent(start_xyz)
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent = tangent / np.linalg.norm(tangent)

        cos_n = np.cos(np.deg2rad(delta_to_normal))
        sin_n = np.sin(np.deg2rad(delta_to_normal))
        vec = cos_n * normal + sin_n * tangent
        forward_dir = vec / np.linalg.norm(vec)  # unit vector

        return start_xyz, forward_dir

    def sample_next_point(
        self, current_xyz, current_forward_dir, d, delta_to_normal_range
    ):
        next_point_xyz = current_xyz + current_forward_dir * d

        next_point_normal = self.get_normal(next_point_xyz)
        next_point_normal = next_point_normal / np.linalg.norm(next_point_normal)

        next_point_tangent = self.get_tangent(next_point_xyz)
        next_point_tangent = (
            next_point_tangent
            - np.dot(next_point_tangent, next_point_normal) * next_point_normal
        )
        next_point_tangent = next_point_tangent / np.linalg.norm(next_point_tangent)

        delta_to_normal = np.random.uniform(
            delta_to_normal_range[0], delta_to_normal_range[1]
        )
        cos_n = np.cos(np.deg2rad(delta_to_normal))
        sin_n = np.sin(np.deg2rad(delta_to_normal))
        vec = cos_n * next_point_normal + sin_n * next_point_tangent
        forward_dir = vec / np.linalg.norm(vec)  # unit vector

        return next_point_xyz, forward_dir
