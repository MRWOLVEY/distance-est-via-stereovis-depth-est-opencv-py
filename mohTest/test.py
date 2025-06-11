# import cv2
# import subprocess
# import time
# import multiprocessing

# def stream_camera(cam_index, stream_name, server_ip, width=640, height=480, fps=30):
#     cap = cv2.VideoCapture(cam_index)

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     cap.set(cv2.CAP_PROP_FPS, fps)

#     if not cap.isOpened():
#         print(f"Camera {cam_index} failed to open.")
#         return

#     rtsp_url = f'rtsp://{server_ip}:8554/{stream_name}'

#     ffmpeg_cmd = [
#         'ffmpeg',
#         '-loglevel', 'error',
#         '-f', 'rawvideo',
#         '-pix_fmt', 'bgr24',
#         '-s', f'{width}x{height}',
#         '-r', str(fps),
#         '-i', '-',  # stdin
#         '-an',
#         '-c:v', 'libx264',
#         '-preset', 'veryfast',
#         '-tune', 'zerolatency',
#         '-f', 'rtsp',
#         rtsp_url
#     ]

#     print(f"[Camera {cam_index}] Streaming to {rtsp_url} ...")
#     process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"[Camera {cam_index}] Frame read failed.")
#                 break
#             process.stdin.write(frame.tobytes())
#     except KeyboardInterrupt:
#         print(f"[Camera {cam_index}] Stopped by user.")
#     except Exception as e:
#         print(f"[Camera {cam_index}] Error: {e}")
#     finally:
#         cap.release()
#         process.stdin.close()
#         process.wait()
#         print(f"[Camera {cam_index}] Streaming ended.")

# if __name__ == '__main__':
#     # Replace with your MediaMTX server's IP
#     SERVER_IP = '192.168.119.173'

#     cam_configs = [
#         (0, 'cam1'),
#         (1, 'cam2')
#     ]

#     processes = []

#     for cam_index, stream_name in cam_configs:
#         p = multiprocessing.Process(target=stream_camera, args=(cam_index, stream_name, SERVER_IP))
#         p.start()
#         processes.append(p)

#     try:
#         for p in processes:
#             p.join()
#     except KeyboardInterrupt:
#         print("Shutting down all streams...")
#         for p in processes:
#             p.terminate()
#####################################
import cv2
import subprocess
import time
import multiprocessing

def stream_camera(cam_index, stream_name, width=640, height=480, fps=15):
    cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"Camera {cam_index} failed to open.")
        return

    rtsp_url = f'rtsp://localhost:8554/{stream_name}'

    ffmpeg_cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',  # stdin
        '-an',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-f', 'rtsp',
        rtsp_url
    ]

    print(f"[Camera {cam_index}] Starting FFmpeg to stream to {rtsp_url}...")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[Camera {cam_index}] Frame read failed.")
                break
            process.stdin.write(frame.tobytes())
    except KeyboardInterrupt:
        print(f"[Camera {cam_index}] Stopped by user.")
    except Exception as e:
        print(f"[Camera {cam_index}] Error: {e}")
    finally:
        cap.release()
        process.stdin.close()
        process.wait()
        print(f"[Camera {cam_index}] Streaming ended.")

if __name__ == '__main__':
    cam_configs = [
        (1, 'cam1'),  # e.g. Pi cam
        (2, 'cam2')   # e.g. USB cam
    ]

    processes = []

    for cam_index, stream_name in cam_configs:
        p = multiprocessing.Process(target=stream_camera, args=(cam_index, stream_name))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Shutting down all streams...")
        for p in processes:
           p.terminate()
###########################################
# import cv2
# import subprocess
# import threading
# import time

# def stream_camera(cam_index, stream_name):
#     width, height, fps = 640, 480, 25
#     cap = cv2.VideoCapture(cam_index)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
#     ffmpeg_cmd = [
#         'ffmpeg',
#         '-f', 'rawvideo',
#         '-pix_fmt', 'bgr24',
#         '-s', f'{width}x{height}',
#         '-r', str(fps),
#         '-i', '-',
#         '-an',
#         '-vcodec', 'libx264',
#         '-preset', 'ultrafast',
#         '-f', 'rtsp',
#         f'rtsp://<ASP-SERVER-IP>:8554/{stream_name}'
#     ]

#     process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#     print(f"[{stream_name}] Streaming started")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         process.stdin.write(frame.tobytes())

# # Run both cameras in parallel threads
# t1 = threading.Thread(target=stream_camera, args=(2, "left"))
# t2 = threading.Thread(target=stream_camera, args=(1, "right"))

# t1.start()
# t2.start()
