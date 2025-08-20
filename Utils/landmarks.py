import cv2 as cv
import numpy as np

class Landmark:
    @staticmethod
    def draw_landmarks(frame, pts, connect=True):
        """pts shape: (68, 2)"""
        for (x, y) in pts:
            cv.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        if not connect:
            return

        # Landmark index groups
        jaw = list(range(0, 17))
        rb  = list(range(17, 22))
        lb  = list(range(22, 27))
        nb  = list(range(27, 31))
        nl  = list(range(31, 36))
        re  = list(range(36, 42))
        le  = list(range(42, 48))
        ol  = list(range(48, 60))
        il  = list(range(60, 68))

        # Use static helper
        Landmark.poly(frame, pts, jaw)
        Landmark.poly(frame, pts, rb)
        Landmark.poly(frame, pts, lb)
        Landmark.poly(frame, pts, nb)
        Landmark.poly(frame, pts, nl, closed=True)
        Landmark.poly(frame, pts, re, closed=True)
        Landmark.poly(frame, pts, le, closed=True)
        Landmark.poly(frame, pts, ol, closed=True)
        Landmark.poly(frame, pts, il, closed=True)

    @staticmethod
    def poly(frame, pts, idxs, closed=False):
        """Draw polyline connecting given landmark indices"""
        p = pts[idxs]
        for i in range(len(p) - 1):
            cv.line(frame,
                    tuple(p[i].astype(int)),
                    tuple(p[i+1].astype(int)),
                    (255, 200, 0), 1)
        if closed:
            cv.line(frame,
                    tuple(p[-1].astype(int)),
                    tuple(p[0].astype(int)),
                    (255, 200, 0), 1)
