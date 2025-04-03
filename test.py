import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("1_.JPG")
img2 = cv.imread("2_.JPG")

#display images using plt subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img1)
axes[0].set_title("Left Image")
axes[1].imshow(img2)
axes[1].set_title("Right Image")
plt.show()