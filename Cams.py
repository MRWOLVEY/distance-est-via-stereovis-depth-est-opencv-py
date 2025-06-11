import cv2
import subprocess
import threading
import time

def stream_camera(cam_index, stream_name):
    width, height, fps = 1280, 720, 30
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-g', str(fps),
        '-b:v', '500K',
        '-bufsize', '2M',
        '-f', 'rtsp',
        f'rtsp://127.0.0.1:8554/{stream_name}'
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    print(f"[{stream_name}] Streaming started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process.stdin.write(frame.tobytes())

# Run both cameras in parallel threads
t1 = threading.Thread(target=stream_camera, args=(2, "left"))
t2 = threading.Thread(target=stream_camera, args=(1, "right"))

t1.start()
t2.start()

t1.join()
t2.join()

