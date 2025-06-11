import cv2
import numpy as np
import glob
import os

def calibrate_camera(images_dir, chessboard_size=(9,6), square_size=1.0):
    """
    Calibrate a single camera using chessboard images.

    Args:
        images_dir (str): path to folder of calibration images.
        chessboard_size (tuple): number of inner corners per a chessboard row and column (cols, rows).
        square_size (float): size of one square in your defined unit (e.g., meters).

    Returns:
        ret (float): reprojection error.
        camera_matrix (ndarray): 3×3 intrinsic matrix.
        dist_coeffs (ndarray): distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6]]).
        rvecs, tvecs: extrinsic parameters for each image.
    """
    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                30,  # max 30 iterations
                1e-6)

    # Prepare object points: (0,0,0), (1,0,0), ... in 3D space
    objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
    objp[:,:2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Load images
    images = glob.glob(os.path.join(images_dir, '*.jpg'))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {images_dir}")
    n = 0
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: couldn't read {fname}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, refine and add points
        if ret:
            objpoints.append(objp)

            corners_refined = cv2.cornerSubPix(
                gray, corners, winSize=(11,11), zeroZone=(-1,-1), criteria=criteria
            )
            imgpoints.append(corners_refined)

            # Optional: draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners_refined, ret)
            cv2.imshow('Corners', img)
            cv2.imwrite(f'corner{n}.jpg', img)
            n = n + 1
            cv2.waitKey(1000)
        else:
            print(f"Chessboard not detected in {fname}")

    cv2.destroyAllWindows()

    # Calibrate the camera now
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

if __name__ == "__main__":
    # Path to your folder of chessboard images
    images_folder = "./"

    # Number of inner corners per chessboard row and column
    # e.g., a 9×6 board has 9 inner corners along width, 6 along height
    board_size = (8, 6)

    # Size of each square (in your chosen unit: meters, centimeters, etc.)
    square_dim = 0.025  # e.g. 2.5 cm squares

    print("Calibrating camera...")
    error, K, dist, rvecs, tvecs = calibrate_camera(
        images_folder, chessboard_size=board_size, square_size=square_dim
    )

    print("\n=== Calibration Results ===")
    print(f"Reprojection Error: {error:.4f}")
    print("Camera intrinsic matrix (K):")
    print(K)
    print("\nDistortion coefficients (k1,k2,p1,p2[,k3…]):")
    print(dist)

    # np.savez("./results/gpt_calibration_data.npz",
    #      camera_matrix=K,
    #      dist_coeffs=dist,
    #      rvecs=rvecs,
    #      tvecs=tvecs)
