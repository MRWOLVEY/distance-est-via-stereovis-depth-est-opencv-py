import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images
img1 = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)  # Left image
img2 = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)  # Right image

"""Camera Calibration: we need a fu*king chessboard. :("""

""" """

if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Check the file paths.")

# Load images
# img1 = cv2.imread("left.jpg")
# img2 = cv2.imread("right.jpg")

"""according to chatGPT, my phone's cam's params are as follows:"""
fx = 2967
cx = 2048
fy = 2967
cy = 1536

k1 = -0.25
k2 = 0.10
p1 = 0.0
p2 = 0.0
k3 = -0.05

# Load camera parameters (replace with actual values)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # Intrinsic params
dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Distortion coefficients

# Stereo rectification
baseline = 0.06
R = np.eye(3)  # Assuming no rotation (modify if needed)
T = np.array([[baseline], [0], [0]])  # Baseline along X-axis

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, img1.shape[:2], R, T
)

# Rectify images
map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, img1.shape[:2], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, img2.shape[:2], cv2.CV_32FC1)

rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

cv2.imwrite("rectified_left.jpg", rectified1)
cv2.imwrite("rectified_right.jpg", rectified2)


# Stereo Block Matching (SBM) for disparity map computation
stereo = cv2.StereoSGBM_create(
    minDisparity=0,  # Minimum possible disparity
    numDisparities=64,  # Must be divisible by 16
    blockSize=9,  # Matched block size (odd number)
    P1=8 * 3 * 9**2,  # Regularization term for smoothness (empirical tuning)
    P2=32 * 3 * 9**2,  # Stronger regularization
    disp12MaxDiff=1,  # Max allowed difference between left-right disparity
    uniquenessRatio=10,  # Reject weak matches
    speckleWindowSize=100,  # Filter out noise
    speckleRange=32,  # Ignore large disparity jumps
)

# Compute the disparity map
disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0  # Normalize disparity

# Normalize for visualization
disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_visual = np.uint8(disparity_visual)

# Plot the images and depth map
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img1, cmap="gray")
axes[0].set_title("Left Image")
axes[0].axis("off")

axes[1].imshow(img2, cmap="gray")
axes[1].set_title("Right Image")
axes[1].axis("off")

axes[2].imshow(disparity_visual, cmap="gray")
axes[2].set_title("Depth (Disparity Map)")
axes[2].axis("off")

plt.show()
