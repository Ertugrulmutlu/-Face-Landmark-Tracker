# Utils/helpers.py
import os, csv
import cv2 as cv
import numpy as np
from Utils.landmarks import Landmark

class Helpers:
    """
    Misc helpers for I/O, CSV, and per-frame processing (face -> pad -> facemark -> smooth -> draw).
    """

    # --- EMA smoothing state (single face) ---
    _prev_pts = None     # (68,2) float for EMA
    _alpha    = 0.7      # default EMA alpha (0.6–0.9)

    # --- caches for SKIP/hold ---
    _last_landmarks = None   # list of (68,2) arrays (in case of multi-face later)

    # ------------- FS / I/O -------------
    @staticmethod
    def ensure_dirs(output_video, csv_path=None, save_csv=False):
        ov_dir = os.path.dirname(output_video)
        if ov_dir:
            os.makedirs(ov_dir, exist_ok=True)
        if save_csv and csv_path:
            cp_dir = os.path.dirname(csv_path)
            if cp_dir:
                os.makedirs(cp_dir, exist_ok=True)

    @staticmethod
    def check_files(paths):
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")

    @staticmethod
    def open_video(input_path, output_path):
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open input video.")

        fps = cap.get(cv.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        writer = None
        for fourcc_str in ("mp4v", "avc1", "XVID"):
            writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*fourcc_str), fps, (w, h))
            if writer.isOpened():
                break
        if not writer or not writer.isOpened():
            raise RuntimeError("Cannot open VideoWriter with common codecs (mp4v/avc1/XVID).")

        return cap, writer, fps, w, h

    @staticmethod
    def init_csv(csv_path):
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        header = ["frame_idx"] + [f"x{i}" for i in range(68)] + [f"y{i}" for i in range(68)]
        writer.writerow(header)
        return csv_file, writer

    # ------------- Geometry -------------
    @staticmethod
    def pad_box(x, y, w, h, scale, W, H):
        cx, cy = x + w/2.0, y + h/2.0
        nw, nh = w * scale, h * scale
        nx, ny = int(max(0, cx - nw/2.0)), int(max(0, cy - nh/2.0))
        nw, nh = int(min(nw, W - nx)), int(min(nh, H - ny))
        return nx, ny, nw, nh

    # ------------- Smoothing control -------------
    @staticmethod
    def set_smoothing(alpha: float = 0.7):
        Helpers._alpha = float(alpha)

    @staticmethod
    def reset_smoothing():
        Helpers._prev_pts = None

    # ------------- Per-frame pipeline -------------
    @staticmethod
    def process_frame(frame,
                      frame_idx,
                      det,
                      facemark,
                      conf_thresh: float,
                      max_faces: int,
                      csv_writer=None,
                      save_csv: bool = False,
                      pad_scale: float = 1.25,
                      draw_boxes: bool = True,
                      connect: bool = True,
                      skip: bool = False,
                      alpha: float = None):
        """
        If skip=True: don't run detection; redraw last known landmarks on this frame.
        Otherwise: detect -> pad boxes -> facemark -> EMA smooth -> draw -> (optional) CSV/boxes.
        """
        # --- SKIP path: redraw last landmarks to keep video/overlay in sync ---
        if skip:
            if Helpers._last_landmarks is not None:
                for pts in Helpers._last_landmarks:
                    Landmark.draw_landmarks(frame, pts, connect=connect)
            return frame

        # --- 1) detect faces (DNN) ---
        boxes = det.detect_faces_dnn(frame, conf_thresh)

        # --- 2) keep largest N ---
        if len(boxes) > max_faces:
            boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)[:max_faces]

        if not boxes:
            # no face → reset smoothing and clear last cache
            Helpers._prev_pts = None
            Helpers._last_landmarks = None
            return frame

        # --- 3) pad boxes ---
        H, W = frame.shape[:2]
        boxes = [Helpers.pad_box(x, y, bw, bh, pad_scale, W, H) for (x, y, bw, bh) in boxes]

        # --- 4) facemark landmarks ---
        try:
            ok, landmarks = facemark.fit(frame, np.array(boxes))
            if not ok or landmarks is None or len(landmarks) == 0:
                return frame
        except Exception:
            return frame

        # --- 5) EMA smoothing (single face varsayımı) & draw ---
        a = Helpers._alpha if alpha is None else float(alpha)
        smoothed_list = []
        for lm in landmarks:
            pts = lm[0]  # (68,2)

            if Helpers._prev_pts is None:
                smooth = pts.copy()
            else:
                smooth = a * Helpers._prev_pts + (1.0 - a) * pts

            Helpers._prev_pts = smooth
            smoothed_list.append(smooth)

            Landmark.draw_landmarks(frame, smooth, connect=connect)

            if save_csv and csv_writer:
                row = [frame_idx] + [float(p[0]) for p in smooth] + [float(p[1]) for p in smooth]
                csv_writer.writerow(row)

        # cache for SKIP frames
        Helpers._last_landmarks = smoothed_list

        # --- 6) draw boxes (optional) ---
        if draw_boxes:
            for (x, y, bw, bh) in boxes:
                cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 150, 255), 1)

        return frame
