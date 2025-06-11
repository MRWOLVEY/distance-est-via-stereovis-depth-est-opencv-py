import cv2 as cv

cap1 = cv.VideoCapture(2)  # Change to the appropriate camera index if needed
cap2 = cv.VideoCapture(1)  # Change to the appropriate camera index if needed

cap1.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cap2.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()

cv.imwrite("left.png", frame1)
cv.imwrite("right.png", frame2)
cap1.release()
cap2.release()
cv.destroyAllWindows()
