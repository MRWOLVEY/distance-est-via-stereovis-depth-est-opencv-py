import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def draw_center_lines(frame):
    height, width = frame.shape[:2]
    # Draw vertical center line
    # cv.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
    
    cv.line(frame, (134, 0), (134, height - 1), (0, 0, 0), 2)
    cv.line(frame, (503, 0), (503, height - 1), (0, 255, 0), 2)
    cv.line(frame, (872, 0), (872, height - 1), (0, 255, 0), 2)
    cv.line(frame, (1241, 0), (1241, height - 1), (0, 0, 0), 2)
    # Draw horizontal center line
    # cv.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

def draw_center_lines2(frame):
    height, width = frame.shape[:2]
    # Draw vertical center line
    # cv.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

    cv.line(frame, (0, 0), (0, height - 1), (0, 0, 0), 2)
    cv.line(frame, (369, 0), (369, height - 1), (0, 255, 0), 2)
    cv.line(frame, (738, 0), (738, height - 1), (0, 255, 0), 2)
    cv.line(frame, (1107, 0), (1107, height - 1), (0, 0, 0), 2)
    
    # Draw horizontal center line
    # cv.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

def main():
    # Open video capture for two cameras
    cap1 = cv.VideoCapture(2)  # Change to the appropriate camera index if needed
    # cap1 = cv.VideoCapture(1)  # Change to the appropriate camera index if needed
    cap2 = cv.VideoCapture(0)  # Change to the appropriate camera index if needed
    # cap2 = cv.VideoCapture(2)  # Change to the appropriate camera index if needed

    width, height, fps = 1280, 720, 25
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    cap2.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    while True:

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to capture video from one of the cameras.")
            break

        # Draw center lines on both frames
        draw_center_lines(frame1)
        draw_center_lines2(frame2)

        plt.subplot(1,2,1)
        plt.imshow(frame1)
        plt.subplot(1,2,2)
        plt.imshow(frame2)
        # plt.show()
        # Display the frames
        cv.imshow('left', frame2)
        cv.imshow('right', frame1)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # # Release the video captures and close windows
    # cap1.release()
    # cap2.release()
    # cv.destroyAllWindows()

if __name__ == "__main__":
    main()