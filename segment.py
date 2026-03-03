import os
import glob
import argparse
import numpy as np
import cv2
import torch
from typing import List, Optional, Tuple
from sam2.build_sam import build_sam2_camera_predictor
import imageio.v2 as imageio
import json

# Mouse interaction globals
click_point = None


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = [x, y]
        print(f"Point selected: ({x}, {y})")
        print("Point selected! Press any key to continue...")


def draw_point_on_frame(frame, click_point=None):
    frame_copy = frame.copy()
    if click_point is not None:
        cv2.circle(frame_copy, tuple(click_point), 8, (0, 255, 0), -1)
        cv2.circle(frame_copy, tuple(click_point), 12, (0, 255, 0), 2)
    return frame_copy


def reset_point_selection():
    global click_point
    click_point = None


def overlay_mask_on_image(image, mask, color, alpha=0.5):
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    return overlay.astype(np.uint8)


def overlay_masks_with_overlap(
    image: np.ndarray,
    knife_mask: Optional[np.ndarray],
    object_mask: Optional[np.ndarray],
    knife_color=(0, 255, 0),
    object_color=(0, 0, 255),
    overlap_color=(255, 255, 0),
    alpha=0.5,
) -> np.ndarray:
    vis = image.copy()
    if knife_mask is not None:
        vis = overlay_mask_on_image(vis, knife_mask, knife_color, alpha)
    if object_mask is not None:
        vis = overlay_mask_on_image(vis, object_mask, object_color, alpha)
    if (knife_mask is not None) and (object_mask is not None):
        overlap = (knife_mask.astype(bool) & object_mask.astype(bool)).astype(np.uint8)
        vis = overlay_mask_on_image(vis, overlap, overlap_color, alpha)
    return vis


def interactive_select_points(
    frames: List[str],
    model_cfg: str,
    sam2_checkpoint: str,
    object_name: str,
    obj_id: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    init_frame_idx: int = 0,
) -> Optional[Tuple[int, np.ndarray]]:
    """
    Interactive multi-point selection with preview.
    Returns the actual frame index used for segmentation and selected points.
    """
    global click_point
    if not frames:
        return None

    current_idx = init_frame_idx
    points: List[Tuple[int, int]] = []
    labels: List[int] = []

    window_name = f"Select {object_name} (click points, y=accept, n=reset, <-/-> change frame, Esc=cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    preview_mask: Optional[np.ndarray] = None
    predictor_preview = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    while True:
        bgr = cv2.imread(frames[current_idx])
        if bgr is None:
            print(f"[Warn] Failed to read: {frames[current_idx]}")
            current_idx = (current_idx + 1) % len(frames)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        display = rgb.copy()
        # Draw selected points
        for px, py in points:
            cv2.circle(display, (int(px), int(py)), 6, color, -1)
            cv2.circle(display, (int(px), int(py)), 10, color, 2)

        # overlay preview mask
        if preview_mask is not None:
            display = overlay_mask_on_image(display, preview_mask, color, alpha=0.4)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if click_point is not None:
            points.append((click_point[0], click_point[1]))
            labels.append(1)
            click_point = None
            # generate preview mask
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictor_preview.load_first_frame(rgb)
                    pts_np = np.array(points, dtype=np.float32)
                    lbs_np = np.array(labels, dtype=np.int32)
                    predictor_preview.add_new_prompt(
                        frame_idx=0, obj_id=obj_id, points=pts_np, labels=lbs_np
                    )
                    out_ids, out_logits = predictor_preview.track(rgb)
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
                print(f"[Warn] Preview failed: {e}")

        if key == ord("y"):
            cv2.destroyWindow(window_name)
            return current_idx, np.array(points, dtype=np.float32)
        elif key == ord("n"):
            points = []
            labels = []
            preview_mask = None
        elif key == 27:  # Esc
            cv2.destroyWindow(window_name)
            return None
        elif key == 81:  # Left arrow
            current_idx = (current_idx - 1) % len(frames)
            print(f"Current frame: {current_idx}")
            points = []
            labels = []
            preview_mask = None
        elif key == 83 or key == 80:  # Right arrow
            current_idx = (current_idx + 1) % len(frames)
            print(f"Current frame: {current_idx}")
            points = []
            labels = []
            preview_mask = None


def interactive_select_object_points(
    frames: List[str], model_cfg: str, sam2_checkpoint: str, init_frame_idx: int = 0
) -> Optional[Tuple[int, np.ndarray]]:
    return interactive_select_points(
        frames, model_cfg, sam2_checkpoint, "Object", 2, (0, 0, 255), init_frame_idx
    )


def interactive_select_knife_points(
    frames: List[str], model_cfg: str, sam2_checkpoint: str, init_frame_idx: int = 0
) -> Optional[Tuple[int, np.ndarray]]:
    return interactive_select_points(
        frames, model_cfg, sam2_checkpoint, "Knife", 1, (0, 255, 0), init_frame_idx
    )


def find_camera_rgb_frames(
    camera_type: str, timestamp_dir: str, patterns: List[str]
) -> List[str]:
    base = os.path.join(timestamp_dir, f"{camera_type}_rgb")
    results: List[str] = []
    if os.path.isdir(base):
        for pat in patterns:
            results.extend(sorted(glob.glob(os.path.join(base, pat))))

    if results:
        import re

        def extract_index(path: str) -> Tuple[int, str]:
            name = os.path.splitext(os.path.basename(path))[0]
            nums = re.findall(r"\d+", name)
            if nums:
                return int(nums[-1]), name
            return float("inf"), name

        results.sort(key=lambda p: extract_index(p))

    return results


def process_timestamp(
    camera_type,
    timestamp_dir,
    target_images,
    model_cfg,
    sam2_checkpoint,
    show_live=False,
    knife_idx=None,
    knife_points=None,
    obj_idx=None,
    obj_points=None,
):
    knife_dir = os.path.join(timestamp_dir, f"{camera_type}_seg_knife")
    object_dir = os.path.join(timestamp_dir, f"{camera_type}_seg_object")
    overlay_dir = os.path.join(timestamp_dir, f"{camera_type}_seg_overlay")
    os.makedirs(knife_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # --- Interactive selection or use provided selections ---
    if knife_idx is None or knife_points is None:
        knife_selection = interactive_select_knife_points(
            target_images, model_cfg, sam2_checkpoint
        )
        if not knife_selection:
            print("No knife points selected, skipping timestamp:", timestamp_dir)
            return
        knife_idx, knife_points = knife_selection
    knife_points = (
        np.array(knife_points, dtype=np.float32) if knife_points is not None else None
    )
    knife_labels = (
        np.ones(len(knife_points), dtype=np.int32) if knife_points is not None else None
    )

    if obj_idx is None or obj_points is None:
        obj_selection = interactive_select_object_points(
            target_images, model_cfg, sam2_checkpoint, init_frame_idx=knife_idx
        )
        if obj_selection:
            obj_idx, obj_points = obj_selection
        else:
            obj_idx, obj_points = -1, None
    obj_points = (
        np.array(obj_points, dtype=np.float32) if obj_points is not None else None
    )
    obj_labels = (
        np.ones(len(obj_points), dtype=np.int32) if obj_points is not None else None
    )
    if obj_points is None:
        print("No object points selected.")

    # --- Main predictor ---
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        start_idx = min(knife_idx, obj_idx if obj_idx >= 0 else knife_idx)
        start_rgb = cv2.cvtColor(
            cv2.imread(target_images[start_idx]), cv2.COLOR_BGR2RGB
        )
        predictor.load_first_frame(start_rgb)

        if knife_points is not None and knife_idx == start_idx:
            predictor.add_new_prompt(start_idx, 1, knife_points, knife_labels)
        if obj_points is not None and obj_idx == start_idx:
            predictor.add_new_prompt(start_idx, 2, obj_points, obj_labels)

        # --- Process frames sequentially ---
        for idx, path in enumerate(target_images):
            if idx < start_idx:
                continue

            bgr = cv2.imread(path)
            if bgr is None:
                print(f"Failed to read {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if knife_points is not None and idx == knife_idx and idx != start_idx:
                predictor.add_new_prompt(idx, 1, knife_points, knife_labels)
            if obj_points is not None and idx == obj_idx and idx != start_idx:
                predictor.add_new_prompt(idx, 2, obj_points, obj_labels)

            # Track masks
            obj_ids, mask_logits = predictor.track(rgb)

            knife_mask = object_mask = None
            for i, obj_id in enumerate(obj_ids):
                mask = (
                    (mask_logits[i] > 0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    .squeeze()
                )
                if obj_id == 1:
                    cv2.imwrite(
                        os.path.join(knife_dir, os.path.basename(path)), mask * 255
                    )
                    knife_mask = mask
                elif obj_id == 2:
                    cv2.imwrite(
                        os.path.join(object_dir, os.path.basename(path)), mask * 255
                    )
                    object_mask = mask

            # Overlay visualization
            if knife_mask is not None or object_mask is not None:
                vis = overlay_masks_with_overlap(rgb, knife_mask, object_mask)
                cv2.imwrite(
                    os.path.join(overlay_dir, f"overlay_{os.path.basename(path)}"), vis
                )
                if show_live:
                    cv2.imshow(f"{camera_type} overlay", vis)
                    cv2.waitKey(1)

                # --- Write to video (imageio) ---
                video_path = os.path.join(timestamp_dir, "overlay.mp4")
                if idx == start_idx:
                    video_writer = imageio.get_writer(video_path, fps=60)
                if "video_writer" in locals():
                    video_writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            print(f"[{idx + 1}/{len(target_images)}] {os.path.basename(path)} done")

        # --- Release video writer ---
        if "video_writer" in locals():
            video_writer.close()


def collect_selections(
    timestamp_dirs,
    camera_type,
    model_cfg,
    sam2_checkpoint,
    selection_file,
    no_skip=True,
):
    if os.path.exists(selection_file) and not no_skip:
        with open(selection_file, "r") as f:
            selections = json.load(f)
    else:
        selections = {}

    for ts_dir in timestamp_dirs:
        frames = find_camera_rgb_frames(
            camera_type, timestamp_dir=ts_dir, patterns=["*.jpg"]
        )
        if not frames:
            continue
        if ts_dir in selections and not no_skip:
            print(f"  Skipped (already selected): {ts_dir}")
            continue

        print(f"\n[Selection] Timestamp: {ts_dir}")
        knife_sel = interactive_select_knife_points(frames, model_cfg, sam2_checkpoint)
        if not knife_sel:
            print("  Skip (no knife points).")
            continue
        knife_idx, knife_points = knife_sel

        obj_sel = interactive_select_object_points(
            frames, model_cfg, sam2_checkpoint, init_frame_idx=knife_idx
        )
        if obj_sel:
            obj_idx, obj_points = obj_sel
        else:
            obj_idx, obj_points = -1, None

        selections[ts_dir] = {
            "knife": {"idx": knife_idx, "points": knife_points.tolist()},
            "object": {
                "idx": obj_idx,
                "points": obj_points.tolist() if obj_points is not None else None,
            },
        }

    with open(selection_file, "w") as f:
        json.dump(selections, f, indent=2)
    print(f"Selections saved to {selection_file}")
    return selections


def run_segmentation(
    timestamp_dirs,
    camera_type,
    model_cfg,
    sam2_checkpoint,
    selection_file,
    show_live=False,
    no_skip=True,
):
    with open(selection_file, "r") as f:
        selections = json.load(f)

    for ts_dir in timestamp_dirs:
        if ts_dir not in selections:
            continue
        frames = find_camera_rgb_frames(
            camera_type, timestamp_dir=ts_dir, patterns=["*.jpg"]
        )
        if not frames:
            continue

        if not no_skip:
            # Ask to skip or re-segment if all masks already exist
            knife_out_dir = os.path.join(ts_dir, f"{camera_type}_seg_knife")
            object_out_dir = os.path.join(ts_dir, f"{camera_type}_seg_object")
            if os.path.isdir(knife_out_dir) and os.path.isdir(object_out_dir):
                all_knife = all(
                    os.path.exists(os.path.join(knife_out_dir, os.path.basename(f)))
                    for f in frames
                )
                all_object = all(
                    os.path.exists(os.path.join(object_out_dir, os.path.basename(f)))
                    for f in frames
                )
                if all_knife and all_object:
                    print("  Masks already exist for this timestamp. Skipped.")
                    continue

        sel = selections[ts_dir]
        process_timestamp(
            camera_type,
            ts_dir,
            frames,
            model_cfg,
            sam2_checkpoint,
            show_live=show_live,
            knife_idx=sel["knife"]["idx"],
            knife_points=np.array(sel["knife"]["points"]),
            obj_idx=sel["object"]["idx"],
            obj_points=np.array(sel["object"]["points"])
            if sel["object"]["points"]
            else None,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="./data/0908B"
    )
    parser.add_argument("--score", type=str, default="4,5,6,7,8,9")
    parser.add_argument("--camera_type", type=str, default="wrist")
    parser.add_argument("--model-cfg", type=str, default="sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--show-live", action="store_true", help="Display live overlays during tracking"
    )
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise NotADirectoryError(f"Root dir not found: {args.root}")

    # Walk two levels: {score}/{timestamp}
    score_dirs = [
        os.path.join(args.root, d)
        for d in sorted(os.listdir(args.root))
        if os.path.isdir(os.path.join(args.root, d)) and d in args.score.split(",")
    ]
    timestamp_dirs: List[str] = []
    for sdir in score_dirs:
        ts_dirs = [
            os.path.join(sdir, d)
            for d in sorted(os.listdir(sdir))
            if os.path.isdir(os.path.join(sdir, d))
        ]
        timestamp_dirs.extend(ts_dirs)

    if not timestamp_dirs:
        print(f"No timestamp subdirectories under: {args.root}")
        return

    print(f"Found {len(timestamp_dirs)} timestamp folders")
    selection_file = os.path.join(args.root, "selections.json")
    # collect_selections(timestamp_dirs, args.camera_type, args.model_cfg, args.checkpoint, selection_file=selection_file,
    #                    no_skip=args.no_skip)
    run_segmentation(
        timestamp_dirs,
        args.camera_type,
        args.model_cfg,
        args.checkpoint,
        show_live=args.show_live,
        selection_file=selection_file,
        no_skip=args.no_skip,
    )

    print("\nAll timestamps processed.")


if __name__ == "__main__":
    main()
