import cv2
import time
import os

def capture_frames(video_source=1, capture_interval=5, total_frames=20, output_dir="captured_frames"):
    """
    Captures frames from a video source every `capture_interval` seconds until `total_frames` are captured.
    Displays the video feed with a countdown timer.

    Parameters:
    - video_source: The index of the video source (default is 0 for the primary camera).
    - capture_interval: Time interval (in seconds) between captures.
    - total_frames: Total number of frames to capture.
    - output_dir: Directory to save the captured frames.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_count = 0
    start_time = time.time()

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Calculate the remaining time for the next capture
        elapsed_time = time.time() - start_time
        countdown = capture_interval - (elapsed_time % capture_interval)

        # Display the countdown timer on the video feed
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Next capture in: {int(countdown)}s", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the video feed
        cv2.imshow("Video Feed", frame)

        # Capture a frame every `capture_interval` seconds
        if elapsed_time // capture_interval > frame_count:
            frame_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"frame_{frame_count}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Captured frame {frame_count}/{total_frames}: {filename}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminating early...")
            break

    # Release the video source and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Capture complete. Program terminated.")

if __name__ == "__main__":
    capture_frames()