"""
CS3IV Kinect Object Recognition — Person A: Depth Pipeline & Segmentation

To run this script, use command line, do not use the IDE run button.
go into pipeline directory and run: 
pdm install    (do not use pdm init, it breaks stuff as we already have a pyproject.toml)
pdm run pipeline_and_vision ..\data\kinect-training-set.mkv 
run command will be different to use the test set and for a different script

==========================================================================
This file handles:
  1. Loading Kinect video frames (depth, RGB, IR)
  2. Depth thresholding to isolate the object of interest
  3. Mask cleanup using morphological operations
  4. Contour detection and bounding box extraction
  5. RGB-to-depth registration with hardcoded calibrated offsets
  6. Standardised 128x128 crop output for both depth and RGB
  7. Interactive dataset saving with pause/step/label controls

Note on RGB vs Depth alignment:
  The green bounding box is drawn on the DEPTH image window.
  The red bounding box is drawn on the COLOUR image window.
  These are two separate camera sensors on the Kinect with different resolutions
  and a small physical separation (~2cm), so they will never look identical on
  screen — they live in different windows at different scales.
  What matters is that the "Cropped RGB" window shows the correct object.
  The RGB-to-depth offset has been calibrated empirically and is hardcoded as
  OFFSET_X = -65, OFFSET_Y = 15 (see constants section below).
  The red box may appear slightly larger than expected because the RGB sensor
  has a wider field of view than the depth sensor, so the same object covers
  proportionally more RGB pixels.

Note on the arm:
  Since there is always exactly one object in frame, the arm holding it will
  merge into the same depth blob. This is fine — save frames only when the
  object is stationary in the centre of the scene (not during entry/exit)
  for the cleanest crops.

Handoff to Person B (Feature Extraction / Classification):
  - Import and call `segment_object()` to get fixed-size depth + RGB crops.
  - Run this script on training videos to save labelled crops to `output_frames/`.
  - RGB-to-depth offsets are hardcoded: OFFSET_X = -65, OFFSET_Y = 15.
    Use these same values in any batch processing script.
  - Saved crop size: STANDARD_SIZE = (128, 128).
  - Folder structure: output_frames/<label>/depth_NNNN.png + rgb_NNNN.png
  - Depth PNGs are normalised uint8 (0=closest, 255=furthest within the depth
    range). Load normally with cv2.imread(). No special flag needed.

Playback controls:
  SPACE           Pause / unpause
  RIGHT ARROW     Step one frame forward (while paused)
  1-9             Select object label (see LABEL_MAP below)
  S               Save current frame crops under selected label
  Q / ESC         Quit

- Angus 
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import typer

from azure_kinect_video_player.playback_wrapper import AzureKinectPlaybackWrapper

# ---------------------------------------------------------------------------
# CONSTANTS — shared with Person B, do not change without updating B's script
# ---------------------------------------------------------------------------

# Depth range (mm) for isolating the object. Objects placed ~10-48 cm from sensor.
MIN_DEPTH = 100
MAX_DEPTH = 480

# Fixed output crop size. Person B must use the same dimensions when loading data.
STANDARD_SIZE = (128, 128)

# Minimum contour area (pixels²) to ignore noise blobs.
MIN_CONTOUR_AREA = 500

# Morphological kernel size for mask noise removal and hole filling.
MORPH_KERNEL_SIZE = (5, 5)

# Root directory for saved labelled crops.
OUTPUT_DIR = Path("output_frames")

# RGB-to-depth registration offsets (pixels), calibrated empirically.
# Derived from slider values: raw 435 → offset -65, raw 515 → offset +15.
# Pass these to segment_object() and to any batch processing script (Person B).
OFFSET_X = -65   # horizontal shift to align RGB crop with depth bounding box
OFFSET_Y = 15    # vertical shift

# Keyboard label map: press the key to set the active label before saving frames.
# One long video is used where different objects appear in sequence — switch label
# each time a new object enters frame. Keys 1-9 cover objects 1-9;
# 0/a/b cover objects 10-12 (q/s/space are reserved controls).
LABEL_MAP = {
    ord("1"): "mug",
    ord("2"): "illuminator",
    ord("3"): "baby",
    ord("4"): "cup-noodle",
    ord("5"): "measuring-tape",
    ord("6"): "dinosaur",
    ord("7"): "Einstein",
    ord("8"): "hour-glass",
    ord("9"): "label-printer",
    ord("0"): "spray",
    ord("a"): "dog",
    ord("b"): "robot",
}

# ---------------------------------------------------------------------------
# CORE SEGMENTATION FUNCTION — importable by Person B for batch processing
# ---------------------------------------------------------------------------

def segment_object(depth_image, colour_image=None, offset_x=0, offset_y=0):
    """
    Segment the primary object from a depth frame and return standardised crops.

    This is the main handoff function for Person B. Given a raw depth frame
    (and optionally an aligned colour frame), it returns fixed-size crops of
    the dominant object suitable for feature extraction.

    Parameters
    ----------
    depth_image : np.ndarray
        Raw uint16 depth frame from the Kinect (values in millimetres).
    colour_image : np.ndarray or None
        Corresponding BGR colour frame (may be a different resolution).
        Pass None to skip RGB cropping.
    offset_x : int
        Horizontal pixel offset to correct RGB-depth misalignment.
        Use the module-level OFFSET_X constant (hardcoded after calibration).
    offset_y : int
        Vertical pixel offset to correct RGB-depth misalignment.
        Use the module-level OFFSET_Y constant.

    Returns
    -------
    cropped_depth : np.ndarray or None
        (128, 128) uint16 depth crop of the object ROI. None if no object found.
    cropped_rgb : np.ndarray or None
        (128, 128) uint8 BGR colour crop of the object ROI.
        None if colour_image is None or no object found.
    bounding_box : tuple or None
        (x, y, w, h) bounding box in depth image pixel coordinates.
        None if no object found.
    """
    # Step 1 — Binary mask: pixels within the depth range become foreground (255)
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    mask[(depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH)] = 255

    # Step 2 — Morphological cleanup:
    #   OPEN  (erode then dilate): removes small isolated noise blobs
    #   CLOSE (dilate then erode): fills small holes inside the object region
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 3 — Find contours and take the largest (the object + arm in shot)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
        return None, None, None  # Too small — likely noise

    # Step 4 — Bounding box and depth crop
    x, y, w, h = cv2.boundingRect(largest_contour)
    depth_roi = depth_image[y:y + h, x:x + w]
    # Keep raw uint16 values so Person B retains actual mm depths as features
    cropped_depth = cv2.resize(depth_roi, STANDARD_SIZE, interpolation=cv2.INTER_NEAREST)

    # Step 5 — Map bounding box into RGB image space and crop
    # The two cameras have different resolutions and a physical offset, so we
    # scale the depth box by the resolution ratio, then apply the manual offset
    # to correct residual misalignment.
    cropped_rgb = None
    if colour_image is not None:
        scale_x = colour_image.shape[1] / depth_image.shape[1]
        scale_y = colour_image.shape[0] / depth_image.shape[0]

        rgb_x = max(0, int(x * scale_x) + offset_x)
        rgb_y = max(0, int(y * scale_y) + offset_y)
        rgb_w = int(w * scale_x)
        rgb_h = int(h * scale_y)

        # Clamp to image boundary
        rgb_x2 = min(colour_image.shape[1], rgb_x + rgb_w)
        rgb_y2 = min(colour_image.shape[0], rgb_y + rgb_h)

        rgb_roi = colour_image[rgb_y:rgb_y2, rgb_x:rgb_x2]
        if rgb_roi.size > 0:
            cropped_rgb = cv2.resize(rgb_roi, STANDARD_SIZE, interpolation=cv2.INTER_LINEAR)

    return cropped_depth, cropped_rgb, (x, y, w, h)


# ---------------------------------------------------------------------------
# DATASET SAVING HELPER
# ---------------------------------------------------------------------------

def save_frame(label, cropped_depth, cropped_rgb, frame_counters):
    """
    Save depth and/or RGB crops under the given label subfolder.

    Parameters
    ----------
    label : str           Object class name, used as subfolder name.
    cropped_depth : np.ndarray or None
    cropped_rgb : np.ndarray or None
    frame_counters : dict Mutable dict {label: int} tracking per-label counts.

    Returns
    -------
    int : The frame index just saved (for display/logging).
    """
    label_dir = OUTPUT_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    count = frame_counters.get(label, 0)

    if cropped_depth is not None:
        # Zero out pixels outside the valid depth range before normalising.
        # Without this, background pixels (large mm values from the wall/table)
        # stretch the normalisation range, compressing the object to near-black.
        masked_depth = cropped_depth.copy()
        masked_depth[(masked_depth < MIN_DEPTH) | (masked_depth > MAX_DEPTH)] = 0
        depth_to_save = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(str(label_dir / f"depth_{count:04d}.png"), depth_to_save)

    if cropped_rgb is not None:
        cv2.imwrite(str(label_dir / f"rgb_{count:04d}.png"), cropped_rgb)

    frame_counters[label] = count + 1
    return count


# ---------------------------------------------------------------------------
# DISPLAY HELPER — renders all overlays onto display copies of each frame
# ---------------------------------------------------------------------------

def render_displays(colour_image, depth_image, ir_image,
                    cropped_rgb, bbox,
                    current_label, total_saved,
                    offset_x, offset_y, paused, frame_index):
    """
    Draw all annotations and update all windows for the current frame.
    Operates only on display copies — never modifies the source image data.
    """
    # --- Depth window ---
    if depth_image is not None:
        # Normalise to 0-255 grayscale for viewing only (Person B gets raw uint16 crops)
        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        if bbox is not None:
            x, y, w, h = bbox
            # Green box = object location in depth image coordinates
            cv2.rectangle(depth_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(depth_vis, current_label,
                        (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(depth_vis, f"Frame: {frame_index}  [{status}]",
                    (8, depth_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Depth", depth_vis)

        # Mask window — shows the cleaned binary mask for reference
        mask_vis = np.zeros(depth_image.shape, dtype=np.uint8)
        mask_vis[(depth_image > MIN_DEPTH) & (depth_image < MAX_DEPTH)] = 255
        kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
        mask_vis = cv2.morphologyEx(mask_vis, cv2.MORPH_OPEN, kernel)
        mask_vis = cv2.morphologyEx(mask_vis, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Mask", mask_vis)

    # --- Colour window ---
    if colour_image is not None:
        colour_vis = colour_image.copy()

        if bbox is not None and depth_image is not None:
            x, y, w, h = bbox
            scale_x = colour_image.shape[1] / depth_image.shape[1]
            scale_y = colour_image.shape[0] / depth_image.shape[0]
            rx = max(0, int(x * scale_x) + offset_x)
            ry = max(0, int(y * scale_y) + offset_y)
            rw = int(w * scale_x)
            rh = int(h * scale_y)
            # Red box = the depth bounding box mapped into RGB image space
            # using the hardcoded calibration offsets (OFFSET_X, OFFSET_Y).
            cv2.rectangle(colour_vis, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)

        hud = f"Label: {current_label}  |  Saved: {total_saved}  |  offset: ({offset_x},{offset_y})"
        cv2.putText(colour_vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if paused:
            cv2.putText(colour_vis, "-- PAUSED --  SPACE=play  ARROW=step  S=save",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("Colour", colour_vis)

    # --- Cropped RGB window — this is what Person B will receive ---
    if cropped_rgb is not None:
        cv2.imshow("Cropped RGB", cropped_rgb)

    # --- IR window ---
    if ir_image is not None:
        cv2.imshow("IR", ir_image)


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

app = typer.Typer()

@app.command()
def app_main(
    video_filename: Path = typer.Argument(..., help="Path to the Kinect .mkv recording"),
    realtime_wait: bool = typer.Option(False, help="Play back in real time (disable for easy scrubbing)"),
    rgb: bool = typer.Option(True, help="Display and process RGB image"),
    depth: bool = typer.Option(True, help="Display and process depth image"),
    ir: bool = typer.Option(True, help="Display IR image"),
):
    """
    Person A main loop: loads Kinect data, segments objects, and saves labelled crops.

    realtime_wait defaults to False so you can pause and step freely without
    the player getting ahead of you. Set --realtime-wait for normal-speed preview.
    """
    video_filename = Path(video_filename)
    playback_wrapper = AzureKinectPlaybackWrapper(
        video_filename, realtime_wait=realtime_wait,
        auto_start=False, rgb=rgb, depth=depth, ir=ir
    )

    # --- Window setup ---
    if rgb:
        cv2.namedWindow("Colour", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Cropped RGB", cv2.WINDOW_NORMAL)
    if depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    # --- Playback state ---
    paused = False      # Whether the video is currently paused
    step_frame = False  # Becomes True for one iteration when right-arrow is pressed
    frame_index = 0     # Running frame counter

    # Cache of the last decoded frame — kept alive so the display updates while paused
    last_colour = None
    last_depth = None
    last_ir = None
    last_cropped_depth = None
    last_cropped_rgb = None
    last_bbox = None

    # --- Dataset collection state ---
    current_label = "unknown"
    frame_counters = {}  # {label: number_of_frames_saved}
    total_saved = 0

    print("\n=== CS3IV Person A — Dataset Collection ===")
    print("Controls:")
    print("  SPACE           Pause / unpause")
    print("  RIGHT ARROW     Step one frame forward (while paused)")
    print("  S               Save current frame under the active label")
    print("  Q / ESC         Quit\n")
    print("Label keys:")
    for k, name in LABEL_MAP.items():
        print(f"  [{chr(k)}] → {name}")
    print(f"\nActive label: {current_label}\n")

    playback_wrapper.start()
    frame_gen = playback_wrapper.grab_frame()

    try:
        while True:
            # -----------------------------------------------------------
            # FRAME ADVANCE LOGIC
            # Playing normally  → always grab next frame from the generator
            # Paused, no step   → reuse cached frame as-is
            # Paused + step     → grab exactly one new frame, then re-pause
            # -----------------------------------------------------------
            if not paused or step_frame:
                try:
                    colour_image, depth_image, ir_image = next(frame_gen)
                except StopIteration:
                    print("End of video.")
                    break

                if colour_image is None and depth_image is None and ir_image is None:
                    print("End of video.")
                    break

                frame_index += 1
                step_frame = False  # Consume the step command

                # Run segmentation using the hardcoded calibration offsets
                cropped_depth, cropped_rgb, bbox = segment_object(
                    depth_image,
                    colour_image if rgb else None,
                    OFFSET_X, OFFSET_Y
                )

                # Update cache
                last_colour, last_depth, last_ir = colour_image, depth_image, ir_image
                last_cropped_depth, last_cropped_rgb, last_bbox = cropped_depth, cropped_rgb, bbox

            else:
                # Paused — reuse the cached frame with the fixed calibration offsets
                colour_image, depth_image, ir_image = last_colour, last_depth, last_ir

                cropped_depth, cropped_rgb, bbox = segment_object(
                    depth_image,
                    colour_image if rgb else None,
                    OFFSET_X, OFFSET_Y
                )
                last_cropped_depth, last_cropped_rgb, last_bbox = cropped_depth, cropped_rgb, bbox

            # -----------------------------------------------------------
            # RENDER
            # -----------------------------------------------------------
            render_displays(
                colour_image, depth_image, ir_image,
                cropped_rgb, bbox,
                current_label, total_saved,
                OFFSET_X, OFFSET_Y, paused, frame_index
            )

            # -----------------------------------------------------------
            # KEYBOARD HANDLING
            # Use a longer wait when paused to avoid spinning the CPU
            # -----------------------------------------------------------
            wait_ms = 100 if paused else 1
            key = cv2.waitKeyEx(wait_ms)

            RIGHT_ARROW_KEYS = {
                83,         # Linux/OpenCV
                2555904,    # Windows OpenCV
            }

            # Quit
            if key == ord("q") or key == 27:
                break

            # Pause / unpause
            elif key == ord(" "):
                paused = not paused
                print(f"{'Paused' if paused else 'Resuming'} at frame {frame_index}")

            # Step one frame forward 
            elif key in RIGHT_ARROW_KEYS:
                if paused:
                    step_frame = True
                # Ignored during normal playback

            # Change active label
            elif key in LABEL_MAP:
                current_label = LABEL_MAP[key]
                print(f"Label → {current_label}")

            # Save current frame
            elif key == ord("s"):
                if last_cropped_depth is None and last_cropped_rgb is None:
                    print("Warning: no object detected — nothing saved.")
                else:
                    idx = save_frame(current_label, last_cropped_depth, last_cropped_rgb, frame_counters)
                    total_saved += 1
                    print(f"Saved [{current_label}] idx={idx}  (total: {total_saved})")

    except KeyboardInterrupt:
        pass

    # --- Exit summary ---
    print("\n=== Session Summary ===")
    print(f"Frames processed : {frame_index}")
    print(f"Frames saved     : {total_saved}")
    for label, count in sorted(frame_counters.items()):
        print(f"  {label:15s}: {count} frame(s)  →  {OUTPUT_DIR / label}/")

    cv2.destroyAllWindows()
    playback_wrapper.stop()
    return 0


if __name__ == "__main__":
    sys.exit(app())