import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# img1 = cv.imread("samples/lt/lt3.jpg")#lt
# img2 = cv.imread("samples/rt/rt3.jpg")#rt
img1 = cv.imread("ltbb.jpg")#lt
img2 = cv.imread("rtbb.jpg")#rt


#draw a grid with equal area squares on both images
def draw_grid(image,step,):
    # Get image dimensions
    height, width = image.shape[:2]

    # Draw vertical lines
    for x in range(0, width, step):
        cv.line(image, (x, 0), (x, height), (0, 255, 0), 1)

    # Draw horizontal lines
    for y in range(0, height, step):
        cv.line(image, (0, y), (width, y), (0, 255, 0), 1)
    cv.line(image, (0, int(image.shape[0] / 2)), (image.shape[1], int(image.shape[0] / 2)), (255, 0, 0), 2)
    cv.line(image, (int(image.shape[1] / 2),0), (int(image.shape[1]/2), int(image.shape[0])), (255, 0, 0), 2)

draw_grid(img1,100)
draw_grid(img2,100)

cv.line(img1,(1000,0),(1000,3000),(0,0,255),2)
cv.line(img1,(2500,0),(2500,3000),(0,0,255),2)
cv.line(img1,(3900,0),(3900,3000),(0,0,255),2)
cv.line(img2,(500,0),(500,3000),(0,0,255),2) 
cv.line(img2,(2000,0),(2000,3000),(0,0,255),2) 
cv.line(img2,(3500,0),(3500,3000),(0,0,255),2) 

#saving the images
# cv.imwrite("lt_grid.jpg",img1)
# cv.imwrite("rt_grid.jpg",img2)


#display images using plt subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
axes[0].set_title("Left Image")
axes[1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
axes[1].set_title("Right Image")
plt.show()