import sys
import cv2
import numpy as np
import time


def find_depth(circle_right, circle_left, frame_right, frame_left, baseline, f):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    

    if width_right != width_left:
        print('Left and right camera frames do not have the same pixel width')
        return None

    x_right = circle_right[0]
    x_left = circle_left[0]
    print("xr", x_right)
    print("xl", x_left)

    # CALCULATE THE DISPARITY:
    disparity = x_left - x_right  # Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline * f) / disparity  # Depth in [cm]

    return abs(zDepth)


