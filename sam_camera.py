from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2_camera_predictor
from termcolor import cprint

COLORS = [
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 0, 0),  # red
    (0, 255, 255),  # yellow
]

SAM2_CONFIG_DIR = {
    "b+": {
        "config_file": "sam2.1_hiera_b+.yaml",
        "checkpoint_file": "sam2.1_hiera_base_plus.pt",
    },
    "l": {
        "config_file": "sam2.1_hiera_l.yaml",
        "checkpoint_file": "sam2.1_hiera_large.pt",
    },
    "s": {
        "config_file": "sam2.1_hiera_s.yaml",
        "checkpoint_file": "sam2.1_hiera_small.pt",
    },
    "t": {
        "config_file": "sam2.1_hiera_t.yaml",
        "checkpoint_file": "sam2.1_hiera_tiny.pt",
    },
}


class Segment:
    def __init__(
        self,
        model_type: str,
        object_names: List[str],
        show_live: bool = False,
        sam2_dir="/home/sam2",
        device="cuda",
    ):
        self.model_type = model_type
        self.device = device

        self.device_type = "cuda" if "cuda" in device else device
        self.device_id = int(device.split(":")[-1]) if ":" in device else 0

        # Initialize Hydra once with the SAM2 config directory
        try:
            if not GlobalHydra.instance().is_initialized():
                initialize_config_dir(
                    config_dir=f"{sam2_dir}/sam2_configs",
                    job_name="sam2_runtime",
                )
        except Exception as e:
            cprint(f"[SegmentCam] Hydra initialization warning: {e}", "yellow")
        if model_type in SAM2_CONFIG_DIR:
            self.predictor = build_sam2_camera_predictor(
                SAM2_CONFIG_DIR[model_type]["config_file"],
                f"{sam2_dir}/checkpoints/{SAM2_CONFIG_DIR[model_type]['checkpoint_file']}",
                device=device,
            )
        else:
            raise ValueError(f"[SegmentCam] Invalid predictor type: {model_type}")

        self.object_names = object_names
        self.object_id_to_name = {
            i: obj_name for i, obj_name in enumerate(self.object_names)
        }
        self.show_live = show_live
        self.click_point = None
        self.initialized = False
        # Create a persistent display window for step/initialize
        self.window_name = "Seg"
        if self.show_live:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.object_points = {obj_name: None for obj_name in self.object_names}
        self.object_labels = {obj_name: None for obj_name in self.object_names}
        self.object_ids = {obj_name: i for i, obj_name in enumerate(self.object_names)}

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = [x, y, 1]
            print(f"Point selected: ({x}, {y})")
            print("Press any key to continue...")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.click_point = [x, y, 0]
            print(f"Negative point selected: ({x}, {y})")
            print("Press any key to continue...")

    @staticmethod
    def overlay_mask_on_image(image, mask, color, alpha=0.5):
        overlay = image.copy()
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
        return overlay.astype(np.uint8)

    @staticmethod
    def overlay_masks_with_overlap(
        rgb,
        object_masks,
        alpha=0.5,
    ) -> np.ndarray:
        vis = rgb.copy()

        obj_count = 0
        for obj_id, obj_mask in enumerate(object_masks.values()):
            if obj_mask is not None:
                vis = Segment.overlay_mask_on_image(
                    vis, obj_mask, COLORS[obj_id], alpha
                )
                obj_count += 1
        if obj_count == 2:
            # get overlap mask
            overlap = Segment.get_mask_intersection(object_masks)
            overlap_color = np.sum(
                [COLORS[i] for i in range(len(object_masks))], axis=0
            )
            vis = Segment.overlay_mask_on_image(vis, overlap, overlap_color, alpha)
        return vis

    @staticmethod
    def get_mask_intersection(masks):
        """Get the intersection of multiple masks.

        Args:
            masks: dict of masks where the key is the object name and the value is the mask (H, W)

        Returns:
            intersection: np.ndarray of intersection mask
        """
        masks_iter = iter(masks.values())
        intersection = np.array(next(masks_iter), dtype=bool)
        for mask in masks_iter:
            intersection = intersection & np.array(mask, dtype=bool)
        intersection = intersection.astype(np.uint8)
        return intersection

    @staticmethod
    def get_mask_union(masks):
        """Get the union of multiple masks.

        Args:
            masks: dict of masks where the key is the object name and the value is the mask (H, W)

        Returns:
            union: np.ndarray of union mask
        """
        masks_iter = iter(masks.values())
        union = np.array(next(masks_iter), dtype=bool)
        for mask in masks_iter:
            union = union | np.array(mask, dtype=bool)
        union = union.astype(np.uint8)
        return union

    def reset(self):
        # Keep existing predictor; only reset state
        self.initialized = False
        self.object_points = {obj_name: None for obj_name in self.object_names}
        self.object_labels = {obj_name: None for obj_name in self.object_names}

    def set_prompts(self, select_points):
        for obj_name, select_points in select_points.items():
            if select_points is not None:
                self.object_points[obj_name] = select_points.astype(np.float32)
                self.object_labels[obj_name] = np.ones(
                    (self.object_points[obj_name].shape[0],), dtype=np.int32
                )
        # force re-initialize on next step after updating prompts
        self.initialized = False

    def select_prompts(self, init_rgb):
        for obj_id, obj_name in enumerate(self.object_names):
            sel = self.interactive_select_points_on_frame(
                init_rgb, obj_name, obj_id, COLORS[obj_id]
            )
            while sel is None or len(sel) == 0:
                sel = self.interactive_select_points_on_frame(
                    init_rgb, obj_name, obj_id, COLORS[obj_id]
                )
            self.object_points[obj_name] = sel.astype(np.float32)
            self.object_labels[obj_name] = np.ones(
                (self.object_points[obj_name].shape[0],), dtype=np.int32
            )
            print(
                f"  {obj_name} points selected: {self.object_points[obj_name].shape[0]}"
            )

        if len(self.object_names) > 1:
            # force re-initialize on next step to track multiple objects
            self.initialized = False
        else:
            self.initialized = True

    def _extract_masks(self, out_obj_ids, out_mask_logits) -> dict:
        object_masks = {obj_name: None for obj_name in self.object_names}
        for i in range(len(out_obj_ids)):
            obj_id = out_obj_ids[i]
            out_mask = (
                (out_mask_logits[i] > 0.0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            out_mask = out_mask.squeeze()
            object_masks[self.object_id_to_name[obj_id]] = out_mask
        return object_masks

    def initialize(self, init_rgb: np.ndarray) -> Tuple[dict, np.ndarray]:
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(self.device_id).major >= 8
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            self.predictor.load_first_frame(init_rgb.copy())

            for obj_name, obj_point in self.object_points.items():
                obj_id = self.object_ids[obj_name]
                print(f"obj_name: {obj_name}, obj_point: {obj_point}")
                print(f"object_labels: {self.object_labels[obj_name]}")
                if obj_point is not None:
                    _ = self.predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=obj_id,
                        points=obj_point,
                        labels=self.object_labels[obj_name],
                    )
            out_obj_ids, out_mask_logits = self.predictor.track(init_rgb.copy())
        masks = self._extract_masks(out_obj_ids, out_mask_logits)
        overlay = Segment.overlay_masks_with_overlap(init_rgb.copy(), masks)
        if self.show_live:
            cv2.imshow(self.window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        self.initialized = True
        return masks, overlay

    def step(self, rgb: np.ndarray) -> Tuple[dict, np.ndarray]:
        if not self.initialized:
            return self.initialize(rgb)
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(self.device_id).major >= 8
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            out_obj_ids, out_mask_logits = self.predictor.track(rgb.copy())
        masks = self._extract_masks(out_obj_ids, out_mask_logits)
        overlay = Segment.overlay_masks_with_overlap(rgb.copy(), masks)
        if self.show_live:
            cv2.imshow(self.window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return masks, overlay

    def interactive_select_points_on_frame(
        self,
        rgb: np.ndarray,
        object_name: str,
        obj_id: int,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> Optional[np.ndarray]:
        points: List[Tuple[int, int]] = []
        labels: List[int] = []
        preview_mask: Optional[np.ndarray] = None

        window_name = (
            f"Select {object_name} (click add points, y=accept, n=reset, Esc=cancel)"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        print(
            f"  [{object_name}] Click multiple points. Press 'y' to accept, 'n' to reset, 'Esc' to cancel."
        )

        while True:
            display = rgb.copy()
            for i, (px, py) in enumerate(points):
                if labels[i] == 1:
                    point_color = color
                else:
                    point_color = np.array(color) * 0.5
                cv2.circle(display, (int(px), int(py)), 6, point_color, -1)
                cv2.circle(display, (int(px), int(py)), 10, point_color, 2)
            if preview_mask is not None:
                display = Segment.overlay_mask_on_image(display, preview_mask, color)

            cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if self.click_point is not None:
                points.append((self.click_point[0], self.click_point[1]))
                labels.append(self.click_point[2])
                try:
                    with torch.autocast(
                        device_type=self.device_type, dtype=torch.bfloat16
                    ):
                        if torch.cuda.get_device_properties(self.device_id).major >= 8:
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
                        self.predictor.load_first_frame(rgb.copy())
                        pts_np = np.array(points, dtype=np.float32)
                        lbs_np = np.array(labels, dtype=np.int32)
                        _ = self.predictor.add_new_prompt(
                            frame_idx=0, obj_id=obj_id, points=pts_np, labels=lbs_np
                        )
                        out_ids, out_logits = self.predictor.track(rgb.copy())
                        preview_mask = None
                        for i in range(len(out_ids)):
                            if out_ids[i] == obj_id:
                                m = (
                                    (out_logits[i] > 0.0)
                                    .permute(1, 2, 0)
                                    .cpu()
                                    .numpy()
                                    .astype(np.uint8)
                                )
                                preview_mask = m.squeeze()
                                break
                except Exception as e:
                    print(f"  [Warn] Preview failed: {e}")
                self.click_point = None

            if key == ord("y"):
                cv2.destroyWindow(window_name)
                return np.array(points, dtype=np.float32)
            elif key == ord("n"):
                points = []
                labels = []
                preview_mask = None
            elif key == 27:  # Esc
                cv2.destroyWindow(window_name)
                return None


class SamCamera:
    def __init__(
        self,
        model_type="b+",
        show_live=False,
        object_names=["knife", "object"],
        device="cuda",
    ):
        self.sam = Segment(
            model_type=model_type,
            show_live=show_live,
            object_names=object_names,
            device=device,
        )

    @staticmethod
    def combine_seg_masks(seg_masks_1, seg_masks_2):
        # mask: 0/1
        mask1 = (seg_masks_1 > 0).astype(np.int8)
        mask2 = (seg_masks_2 > 0).astype(np.int8)

        # mask1:1，mask2:-1, mask1&mask2:-1, not mask1&mask2:0
        combined_mask = np.zeros_like(mask1, dtype=np.int8)
        combined_mask[(mask1 == 1) & (mask2 == 0)] = 1
        combined_mask[(mask1 == 0) & (mask2 == 1)] = -1
        combined_mask[(mask1 == 1) & (mask2 == 1)] = -1

        binary_combined_mask = mask1 | mask2

        return binary_combined_mask, combined_mask

    def init_seg(self, rgb):
        self.sam.select_prompts(rgb)

    def get_seg(self, rgb):
        masks = self.sam.step(rgb)[0]
        binary_mask = Segment.get_mask_union(masks)
        return {
            "binary_mask": binary_mask,
            "masks": masks,
        }

    def reset(self):
        # Clear state so next step will reinitialize prompts/tracking
        self.sam.reset()


if __name__ == "__main__":
    import time

    rgb_lists = [
        [
            cv2.imread(
                f"./data/0922test/1/20250922_175811319386/wrist_rgb/wrist_rgb_{i}.png"
            ),
            cv2.imread(
                f"./data/0922test/1/20250922_175811319386/wrist2_rgb/wrist2_rgb_{i}.png"
            ),
        ]
        for i in range(46)
    ]

    model_types = ["b+", "s", "t"]
    show_live = False

    for model_type in model_types:
        print(f"Testing {model_type}")

        sam_camera = SamCamera(
            show_live=show_live,
            object_names=["knife", "object", "knife2", "object2"],
            device="cuda:0",
            model_type=model_type,
        )

        # concatenate rgb_list and rgb2_list horizontally
        all_rgb = [np.concatenate(rgb_list, axis=1) for rgb_list in rgb_lists]

        sam_camera.init_seg(all_rgb[0])
        start_time = time.perf_counter()

        for i in range(1, len(all_rgb)):
            combined_rgb = np.concatenate(rgb_lists[i], axis=1)  # (H, W * 2, 3)
            seg = sam_camera.get_seg(all_rgb[i])

            # visualize binary mask
            rgb_mask = (
                (seg["binary_mask"].astype(np.float32)[..., None]) * 255
            )  # (H, W, 1)
            rgb_mask = rgb_mask.repeat(3, axis=-1)  # (H, W, 3)
            cv2.imwrite(
                f"test/rgb_mask_{model_type}_{i}.png", rgb_mask.astype(np.uint8)
            )

            if i % 10 == 0:
                print(f"fps: {10 / (time.perf_counter() - start_time)}")
                start_time = time.perf_counter()

        sam_cameras = [
            SamCamera(
                show_live=show_live,
                object_names=["knife", "object"],
                device=f"cuda:{i}",
                model_type=model_type,
            )
            for i in range(2)
        ]

        for i, sam_camera in enumerate(sam_cameras):
            sam_camera.init_seg(rgb_lists[0][i])
        start_time = time.perf_counter()

        for i in range(1, len(rgb_lists)):
            for j, sam_camera in enumerate(sam_cameras):
                seg = sam_camera.get_seg(rgb_lists[i][j])

                # visualize binary mask
                rgb_mask = (
                    (seg["binary_mask"].astype(np.float32)[..., None]) * 255
                )  # (H, W, 1)
                rgb_mask = rgb_mask.repeat(3, axis=-1)  # (H, W, 3)
                cv2.imwrite(
                    f"test/rgb_mask_{model_type}_{i}_{j}.png", rgb_mask.astype(np.uint8)
                )

            if i % 10 == 0:
                print(f"fps: {10 / (time.perf_counter() - start_time)}")
                start_time = time.perf_counter()
