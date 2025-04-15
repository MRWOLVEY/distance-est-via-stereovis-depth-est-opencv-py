import numpy as np

# Load the file
data = np.load("CalibrationMatrix_college_cpt.npz")

# Print what's inside
print("Keys in file:", data.files)

# Access and print specific matrices
camera_matrix = data["Camera_matrix"]
dist_coeffs = data["distCoeff"]
rvecs = data["RotationalV"]
tvecs = data["TranslationV"]

print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)
