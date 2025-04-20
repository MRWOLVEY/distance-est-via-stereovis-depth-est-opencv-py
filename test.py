import cv2 as cv
import numpy as np

def draw_center_lines(frame):
    height, width = frame.shape[:2]
    # Draw vertical center line
    cv.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
    # Draw horizontal center line
    cv.line(frame, (0, height // 2), (width, height // 2), (255, 0, 0), 2)

def main():
    # Open video capture for two cameras
    cap1 = cv.VideoCapture(2)  # Change to the appropriate camera index if needed
    cap2 = cv.VideoCapture(1)  # Change to the appropriate camera index if needed

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to capture video from one of the cameras.")
            break

        # Draw center lines on both frames
        draw_center_lines(frame1)
        draw_center_lines(frame2)

        # Display the frames
        cv.imshow('left', frame1)
        cv.imshow('right', frame2)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video captures and close windows
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()