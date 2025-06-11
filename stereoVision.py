import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo images
img1 = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)  # Left image
img2 = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)  # Right image


if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Check the file paths.")


# Load camera parameters from calibration
camera_matrix = np.load("mediumCalib/results/gpt_calibration_data.npz")["camera_matrix"]
dist_coeffs = np.load("mediumCalib/results/gpt_calibration_data.npz")["dist_coeffs"]

# Stereo rectification
baseline = 9
R = np.eye(3)  # Rotation matrix (identity for no rotation)
T = np.array([[baseline], [0], [0]])  # Translation vector (baseline along x-axis)

# Ensure consistent dtype (float64 is standard for OpenCV calibration)
camera_matrix = camera_matrix.astype(np.float64)
dist_coeffs = dist_coeffs.astype(np.float64)
R = R.astype(np.float64)
T = T.astype(np.float64)


R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    camera_matrix, dist_coeffs, camera_matrix, dist_coeffs, img1.shape[:2], R, T
)

# Rectify images
map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R1, P1, img1.shape[:2], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, R2, P2, img2.shape[:2], cv2.CV_32FC1)

rectified1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

cv2.imshow("rectified_left.jpg", rectified1)
cv2.imshow("rectified_right.jpg", rectified2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Stereo Block Matching (SBM) for disparity map computation
stereo = cv2.StereoSGBM_create(
    minDisparity=0,  # Minimum possible disparity
    numDisparities=16 * 5,  # Must be divisible by 16
    blockSize=5,  # Matched block size (odd number)
    P1=8 * 3 * 9**2,  # Regularization term for smoothness (empirical tuning)
    P2=32 * 3 * 9**2,  # Stronger regularization
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # SGBM mode
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
