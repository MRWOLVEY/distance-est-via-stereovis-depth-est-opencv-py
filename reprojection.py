import cv2
import numpy as np

# Load stereo images
imgL = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# Load calibration data
data = np.load('calib_data.npz')
cameraMatrix1 = data['cameraMatrix1']
distCoeffs1 = data['distCoeffs1']
cameraMatrix2 = data['cameraMatrix2']
distCoeffs2 = data['distCoeffs2']
R = data['R']
T = data['T']
imageSize = imgL.shape[::-1]


"""RECTIFICATION"""
# Compute rectification transforms
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    imageSize, R, T, flags=cv2.CALIB_ZERO_DISPARITY
)

# Compute rectification maps
map1x, map1y = cv2.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1, imageSize, cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2, imageSize, cv2.CV_32FC1
)

# Apply the rectification maps
rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)


"""DISPARITY MAP"""
# Set parameters for SGBM
min_disp = 0
num_disp = 16 * 5  # Must be divisible by 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity
disparity = stereo.compute(rectifiedL, rectifiedR).astype(np.float32) / 16.0

"""REPROJECTION"""
# Reproject to 3D space
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Example: Get depth of pixel (x, y)
x, y = 100, 100
depth = points_3D[y, x][2]  # Z value in meters (if calibrated correctly)
