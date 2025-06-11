import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
from ultralytics import YOLO
import torch

import triangulation as tri

def commence():
    # Open both cameras
    det_counter = 29
    fps = 30
    # cap_right = cv2.VideoCapture('rtsp://192.168.0.108:8554/right')                    
    # cap_left =  cv2.VideoCapture('rtsp://192.168.0.108:8554/left')
    cap_right = cv2.VideoCapture(1)                    
    cap_left =  cv2.VideoCapture(2)
    cap_right.set(cv2.CAP_PROP_FPS, fps)
    cap_left.set(cv2.CAP_PROP_FPS, fps)

    width, height = 640, 480
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    """We need to split the frames into 3 frame, each frame will be processed by a different thread.
    
    WELL, HOW AM I SUPPOSED TO LET THEM ALL BE WORKING A WHILE LOOP ALL AT THE SAME TIME?
    """
    while True:
        if not cap_right.isOpened() or not cap_left.isOpened():
            print("Error: Could not open video capture.")
            return None, None
        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        det_counter += 1

        if not ret_right or not ret_left:
            print("Error: Could not read frames from video capture.")
            break

        if det_counter % 30 == 0:
            model = YOLO("yolo11s.pt")
            model.to('cuda:0')  # Move model to GPU if available
            print(torch.cuda.is_available())
            """
            SPLIT THE FRAMES HERE
            THEN RUN THE newDistModel FUNCTION IN MULTIPLE THREADS
            FOR IT TO WATCH ALL SPOTS SIMULTANEOUSLY
            """
            spot1_frame_right = frame_right[:, :(369//2)]
            spot1_frame_left = frame_left[:, (134//2):(503//2)]
            # cv2.imshow('frameR', spot1_frame_right)
            # cv2.imshow('frameL', spot1_frame_left)
            # cv2.waitKey(5000)
            # spot2_frame_right = frame_right[:, (369//2):(738//2)]
            # spot2_frame_left = frame_left[:, (503//2):(872//2)]
            # cv2.imshow('frameR', spot2_frame_right)
            # cv2.imshow('frameL', spot2_frame_left)
            # cv2.waitKey(5000)
            spot3_frame_right = frame_right[:, (738//2):(1107//2)]
            spot3_frame_left = frame_left[:, (872//2):(1241//2)]
            # cv2.imshow('frameR', spot3_frame_right)
            # cv2.imshow('frameL', spot3_frame_left)
            cv2.waitKey(5000)
            
            distances_spot1 = newDistModel(spot1_frame_right, spot1_frame_left, model)
            # print('SPOT 1 COMPLETE')
            # distances_spot2 = newDistModel(spot2_frame_right, spot2_frame_left, model)
            # print('SPOT 2 COMPLETE')
            # distances_spot3 = newDistModel(spot3_frame_right, spot3_frame_left, model)
            # print('SPOT 3 COMPLETE')
        
            print("Distance between car in spot 1 and persons: ", distances_spot1)
            # print("Distance between car in spot 2 and persons: ", distances_spot2)
            # print("Distance between car in spot 3 and persons: ", distances_spot3)
        cv2.destroyAllWindows()

def newDistModel(frameR, frameL, model):
    K = np.load('./mediumCalib/results/gpt_calibration_data.npz')['camera_matrix']
    dist_coeff = np.load('./mediumCalib/results/gpt_calibration_data.npz')['dist_coeffs']
    B = 9
    f = 30
    alpha = 50
    
    frameR = cv2.undistort(frameR, K, dist_coeff)
    frameL = cv2.undistort(frameL, K, dist_coeff)
    
    detections_right = obj_det(frameR, model)
    detections_left = obj_det(frameL, model)
    
    # DETECTING CARS AND HUMANS AND STORING THEM ACCORDINGLY
    # STARTING WITH RIGHT
    car_exists_right = False
    persons_centers_right = []
    # figuring out the car in the right frame
    for i in range(len(detections_right)):
        id = int(detections_right[i].cls[0].item())
        if id in [7, 2, 5] and car_exists_right == False: # if it was a car
            car_exists_right = True
            car_pos = detections_right[i].xyxy[0].cpu().numpy()
            car_center_right = ((car_pos[0] + car_pos[2]) // 2, (car_pos[1] + car_pos[3]) // 2)
        elif id in [7, 2, 5] and car_exists_right == True:
            print("More than one car detected in the right frame.")
            continue
            # return []
        
        elif id in [67]:
            person_pos = detections_right[i].xyxy[0].cpu().numpy()
            person_center = ((person_pos[0] + person_pos[2]) // 2, (person_pos[1] + person_pos[3]) // 2)
            persons_centers_right.append(person_center)

        else:
            print("RIGHTNot a car, not a person. The id of the detectee: ", id)

    if car_exists_right == False:
        print("no cars detected RIGHT")
        return

        

        
    
    # LEFT
    car_exists_left = False
    persons_centers_left = []
    for i in range(len(detections_left)):
        id = int(detections_left[i].cls[0].item())
        if id in [7, 2, 5] and car_exists_left == False: # was a car and car didn't exist

            car_exists_left = True
            car_pos = detections_left[i].xyxy[0].cpu().numpy()
            car_center_left = ((car_pos[0] + car_pos[2]) // 2, (car_pos[1] + car_pos[3]) // 2)

        elif id in [7, 2, 5] and car_exists_left == True: # was car and car existed

            # raise ValueError("More than one car detected in the right frame.")
            print("More than one car detected in the right frame.")
            # return []
            continue

        elif id in [67]: # was person (car exists or didn't exist yet)
            person_pos = detections_left[i].xyxy[0].cpu().numpy()
            person_center = ((person_pos[0] + person_pos[2]) // 2, (person_pos[1] + person_pos[3]) // 2)
            persons_centers_left.append(person_center)
        else:
            print("LEFTNot a car, not a person. The id of the detectee: ", id)
    
    # if we went through all left detections and still no car ==> return
    if car_exists_left == False:
        print("No cars detected in LEFT")
        return []
            

    # CALCULATING CAR DEPTH AND POSITION

    # print('CAR CENTER BEFORE DEPTH LEFT', car_center_left)
    # print('CAR CENTER BEFORE DEPTH RIGHT', car_center_right)
    car_depth = tri.find_depth(car_center_right, car_center_left, frameR, frameL, B, f, alpha)
    car_position = pixel_to_3d(car_center_right[0], car_center_right[1], car_depth, K)


    # Calculating people's positions and depths
    persons_positions = []
    persons_depth = []

    for i in range(len(persons_centers_right)):
        if len(persons_centers_left) != len(persons_centers_right):
            print('persons aint the same right left')
            return []
        person_depth = tri.find_depth(persons_centers_right[i], persons_centers_left[i], frameR, frameL, B, f, alpha)
        persons_depth.append(person_depth)
        person_position = pixel_to_3d(persons_centers_right[i][0], persons_centers_right[i][1], person_depth, K)
        persons_positions.append(person_position)
    

    # CALCULATING DISTANCES BETWEEN EVERY HUMAN AND THE EXISTING CAR
    distances = []

    for i in range(len(persons_positions)):
        print(f"3D coordinates of object {i}: ({persons_positions[i][0]:.2f}, {persons_positions[i][1]:.2f}, {persons_positions[i][2]:.2f})")

        distance = compute_distance(np.array(car_position), np.array(persons_positions[i]))
        distances.append(distance)

        print('carPos: ', car_position)
        print('personPos: ', person_position)
        print(f"Distance between car and person {i}: {distance:.2f} m")

    return distances      

def demonstrate():
    K = np.load('./mediumCalib/results/calibration_data.npz')['camera_matrix']
    dist_coeff = np.load('./mediumCalib/results/calibration_data.npz')['dist_coeffs']

    api_ref = cv2.CAP_DSHOW # a parameter that I removed from cv2.VideoCapture(camID, api_ref)
    """it printed this for some reason so I removed it:
    [ WARN:0@6.170] global cap.cpp:344 cv::VideoCapture::open VIDEOIO(DSHOW): backend is generally available but can't be used to capture by index
True
Error: Could not open video capture.
    """
    frame_rate = 30
    B = 9
    f = 30
    alpha = 50
    # cap_right = cv2.VideoCapture('rtsp://127.0.0.1:8554/cam2')                    
    # cap_left =  cv2.VideoCapture('rtsp://127.0.0.1:8554/cam1')
    cap_right = cv2.VideoCapture(2)                    
    cap_left =  cv2.VideoCapture(0)
    cap_right.set(cv2.CAP_PROP_FPS, frame_rate)
    cap_left.set(cv2.CAP_PROP_FPS, frame_rate)
    det_counter = 29

    
    # Load YOLOv5 model
    model = YOLO("yolo11s.pt")
    model.to('cuda:0')  # Move model to GPU if available
    print(torch.cuda.is_available())
    while True:

        if not cap_right.isOpened() or not cap_left.isOpened():
            print("Error: Could not open video capture.")
            break

        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()
        # frame_right, frame_left = calib.undistorted(frame_right, frame_left)


        det_counter += 1

        if not ret_right or not ret_left:
            print("Error: Could not read frames from video capture.")
            break

        
        # frame_right = cv2.undistort(frame_right, K, dist_coeff)
        # frame_left = cv2.undistort(frame_left, K, dist_coeff)
        
        # My way of not making the program have to run the model every single frame. Only one per second will suffice.
        if det_counter % 30 == 0:
            detections_right = obj_det(frame_right, model)
            detections_left = obj_det(frame_left, model)
            # print(detections_right)
            # return

        
        ###################uncommented###############################       
        # car_centers_right, human_centers_right = get_centers(detections_right)
        # car_centers_left, human_centers_left = get_centers(detections_left)
        # car_spot1_rt=closest_centers_to_roi(500,2000,car_centers_right)
        # car_spot1_lt=closest_centers_to_roi(1000,2500,car_centers_left)
        # car_spot2_rt=closest_centers_to_roi(2000,3500,car_centers_right)
        # car_spot2_lt=closest_centers_to_roi(2500,4000,car_centers_left)
        # person_spot1_rt=closest_centers_to_roi(500,2000,human_centers_right)
        # person_spot1_lt=closest_centers_to_roi(1000,2500,human_centers_left)
        # person_spot2_rt=closest_centers_to_roi(2000,3500,human_centers_right)
        # person_spot2_lt=closest_centers_to_roi(2500,4000,human_centers_left)
        # spots=[{"p":{'l':person_spot1_lt,'r':person_spot1_rt},"c":{'l':car_spot1_lt,'r':car_spot1_rt}},
        #        {"p":{'l':person_spot2_lt,'r':person_spot2_rt},"c":{'l':car_spot2_lt,'r':car_spot2_rt}}]
        # print(spots)
        ###################/uncommented###############################
        
        centers_right, centers_left = [], []

        detection_id = {
            'car': 2,
            'person': 0,
            'phone': 67,
            'remote': 65,
            'keyboard': 66
        }
        #drawing detection boxes
        for i in range(len(detections_right)):
            x1, y1, x2, y2 = map(int, detections_right[i].xyxy[0])  # Convert coordinates to integers
            center_right = ((x1+x2)//2, (y1+y2)//2)

            detectee_id = int(detections_right[i].cls[0].item())
            if detectee_id in [detection_id['keyboard'], detection_id['remote']]:
                cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                # print(int(detections_right[i].cls[0].item()))
                centers_right.append(center_right)
                cv2.circle(frame_right, center_right, radius=10, color=(0, 255, 0), thickness=2)
                # cv2.putText(frame_right, str(detections_right[i].cls[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for i in range(len(detections_left)):
            # x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, detections_left[i].xyxy[0])  # Convert coordinates to integers
            #create circle at the center filled with radius of 2 color reqd
            center_left = ((x1+x2)//2, (y1+y2)//2)
            detectee_id = int(detections_left[i].cls[0].item())
            if detectee_id in [detection_id['keyboard'], detection_id['remote']]:
                cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                # print(int(detections_right[i].cls[0].item()))
                centers_left.append(center_left)
                cv2.circle(frame_left, center_left, radius=10, color=(0, 255, 0), thickness=2)

                # cv2.putText(frame_left, str(detections_left[i].cls[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print('centers Left', centers_left)
        print('centers_right', centers_right)
        centers_left, centers_right = sorted(centers_left, key=lambda x:x[0]), sorted(centers_right, key=lambda x:x[0])
        depths = list()
        for i in range(min(len(centers_left), len(centers_right))):
            depth = tri.find_depth(centers_right[i], centers_left[i], frame_right, frame_left, B, f, alpha)
            depths.append(depth)
        print(depths)
        
        positions = list()
        for i in range(len(depths)):
            x, y, z = pixel_to_3d(centers_left[i][0], centers_left[i][1], depths[i], K)
            positions.append((x, y, z))
            # print(f"3D coordinates of object {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
        if det_counter % 30 == 0:
            print(positions)
        
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                if i!= j:
                    cv2.line(frame_left, centers_left[i], centers_left[j], (0, 255, 0), 2)
                    print(positions)

                    cv2.putText(frame_right, f'({positions[i][0]:.2f},{positions[i][1]:.2f},{positions[i][2]:.2f})', (centers_right[i][0], centers_right[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    cv2.putText(frame_right, f'({positions[j][0]:.2f},{positions[j][1]:.2f},{positions[j][2]:.2f}', (centers_right[j][0], centers_right[j][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

                    distance = compute_distance(np.array(positions[i]), np.array(positions[j]))
                    # print(f"Distance between object {i} and object {j}: {distance:.2f} m")
                    # distance=0
                    cv2.putText(frame_left, f"Distance: {distance:.2f}", ((centers_left[i][0]+centers_left[j][0])//2, (centers_left[i][1]+centers_left[j][1])//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        #################################################uncommented###############################
        
        
        # Drawing the centers on frames
        # draw_spot_coordinates(spots,frame_left,frame_right)

        # cv2.line(frame_left,(1000,0),(1000,3000),(0,0,255),2)
        # cv2.line(frame_left,(2500,0),(2500,3000),(0,0,255),2)
        # cv2.line(frame_left,(3900,0),(3900,3000),(0,0,255),2)
        # cv2.line(frame_right,(500,0),(500,3000),(0,0,255),2) 
        # cv2.line(frame_right,(2000,0),(2000,3000),(0,0,255),2) 
        # cv2.line(frame_right,(3500,0),(3500,3000),(0,0,255),2) 

        # cv2.putText(frame_left,"spot_1", (1300,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
        # cv2.putText(frame_left,"spot_2", (2800,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
        # cv2.putText(frame_right,"spot_1",(800,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
        # cv2.putText(frame_right,"spot_2",(2300,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)

        # # Calculating depth for objects in each spot
        # depths=[{'p':[],'c':[]} for i in range(len(spots))]
        # for i in range(len(spots)):
        #     p_coords_lt=spots[i]['p']['l']
        #     p_coords_rt=spots[i]['p']['r']
        #     c_coords_lt=spots[i]['c']['l']
        #     c_coords_rt=spots[i]['c']['r']
        #     for j in range(len(p_coords_lt)):
        #         print(p_coords_rt[j],p_coords_lt[j])      
        #         depths[i]['p'].append(tri.find_depth(p_coords_rt[j],p_coords_lt[j],frame_right,frame_left,B,f,alpha))
        #     for j in range(len(c_coords_lt)):         
        #         print(c_coords_rt[j],c_coords_lt[j])   
        #         depths[i]['c'].append(tri.find_depth(c_coords_rt[j],c_coords_lt[j],frame_right,frame_left,B,f,alpha))

        print(depths)
        # draw_depths(spots,depths,frame_left,frame_right)

        # Display the frames with drawings
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(cv2.cvtColor(frame_left,cv2.COLOR_BGR2RGB))
        # axes[0].set_title("Left Image")
        # axes[1].imshow(cv2.cvtColor(frame_right,cv2.COLOR_BGR2RGB))
        # axes[1].set_title("Right Image")
        # plt.show()
        
        cv2.imshow("frame right", frame_right) 
        cv2.imshow("frame left", frame_left)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release and destroy all windows before termination
    cap_right.release()
    cap_left.release()

    cv2.destroyAllWindows()

def obsoleteMain():
    frame_right = cv2.imread('samples/rt/rt3.jpg')
    frame_left = cv2.imread('samples/lt/lt3.jpg')

    frame_rate = 120    #Camera frame rate (maximum at 120 fps)

    B = 30             #Distance between the cameras [cm]
    f = 30             #Camera lense's focal length [mm]
    alpha = 60       #Camera field of view in the horisontal plane [degrees]

    detections_right = obj_det(frame_right)
    detections_left = obj_det(frame_left)
    """
    # Drawing bounding boxes
    for box in detections_right:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Draw bounding boxes for left frame
    for box in detections_left:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
    
    # Display the frames with bounding boxes
     # Resize frames to fit the monitor
    scale_percent = 25  # Adjust this percentage to fit your monitor
    width = int(frame_right.shape[1] * scale_percent / 100)
    height = int(frame_right.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_frame_right = cv2.resize(frame_right, dim, interpolation=cv2.INTER_AREA)
    resized_frame_left = cv2.resize(frame_left, dim, interpolation=cv2.INTER_AREA)

  
    cv2.imshow("Right Frame with Detections", resized_frame_right)
    cv2.imshow("Left Frame with Detections", resized_frame_left)
    """

    # Get the centers of the detected objects, categorized based on ROIs(spots)
    car_centers_right, human_centers_right = get_centers(detections_right)
    car_centers_left, human_centers_left = get_centers(detections_left)
    car_spot1_rt=closest_centers_to_roi(500,2000,car_centers_right)
    car_spot1_lt=closest_centers_to_roi(1000,2500,car_centers_left)
    car_spot2_rt=closest_centers_to_roi(2000,3500,car_centers_right)
    car_spot2_lt=closest_centers_to_roi(2500,4000,car_centers_left)
    person_spot1_rt=closest_centers_to_roi(500,2000,human_centers_right)
    person_spot1_lt=closest_centers_to_roi(1000,2500,human_centers_left)
    person_spot2_rt=closest_centers_to_roi(2000,3500,human_centers_right)
    person_spot2_lt=closest_centers_to_roi(2500,4000,human_centers_left)
    spots=[{"p":{'l':person_spot1_lt,'r':person_spot1_rt},"c":{'l':car_spot1_lt,'r':car_spot1_rt}},
           {"p":{'l':person_spot2_lt,'r':person_spot2_rt},"c":{'l':car_spot2_lt,'r':car_spot2_rt}}]
    # print(spots)


    """
    #conf scores of all detections
    confs_right=[box.conf[0].item() for box in detections_right]
    confs_left=[box.conf[0].item() for box in detections_left]

    print('left frame: \n', 'cars', car_centers_left, '\n', 'humans', human_centers_left, '\n', 'confs', confs_left, '\n')
    print('right frame: \n', 'cars', car_centers_right, '\n', 'humans', human_centers_right, '\n', 'confs', confs_right, '\n')   

    
    # Drawing the centers on frames
    temp_r= car_centers_right + human_centers_right
    temp_l= car_centers_left + human_centers_left

    for (x, y) in temp_r:
        cv2.circle(frame_right, (x, y), radius=10, color=(0, 255, 0), thickness=2)

    for (x, y) in temp_l:
        cv2.circle(frame_left, (x, y), radius=5, color=(0, 255, 0), thickness=2)

    """
    #drawing detection boxes
    for i in range(len(detections_right)):
        x1, y1, x2, y2 = map(int, detections_right[i].xyxy[0])  # Convert coordinates to integers
        cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame_right, str(detections_right[i].cls[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for box in detections_left:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame_left, str(box.cls[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    draw_spot_coordinates(spots,frame_left,frame_right)

    cv2.line(frame_left,(1000,0),(1000,3000),(0,0,255),2)
    cv2.line(frame_left,(2500,0),(2500,3000),(0,0,255),2)
    cv2.line(frame_left,(3900,0),(3900,3000),(0,0,255),2)
    cv2.line(frame_right,(500,0),(500,3000),(0,0,255),2) 
    cv2.line(frame_right,(2000,0),(2000,3000),(0,0,255),2) 
    cv2.line(frame_right,(3500,0),(3500,3000),(0,0,255),2) 

    cv2.putText(frame_left,"spot_1", (1300,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
    cv2.putText(frame_left,"spot_2", (2800,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
    cv2.putText(frame_right,"spot_1",(800,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
    cv2.putText(frame_right,"spot_2",(2300,300),cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)

    

    #saving both frames
    # cv2.imwrite("frame_left.jpg", frame_left)
    # cv2.imwrite("frame_right.jpg", frame_right)

    # return

        ################## CALCULATING DEPTH ##################
    """
    car_depths = tri.find_depth(car_centers_right, car_centers_left, frame_right, frame_left, B, f, alpha)
    human_depths = tri.find_depth(human_centers_right, human_centers_left, frame_right, frame_left, B, f, alpha)
    print("Car depths: ", car_depths)
    print("Human depths: ", human_depths)
    """
    # Calculating depth for objects in each spot
    depths=[{'p':[],'c':[]} for i in range(len(spots))]
    for i in range(len(spots)):
        p_coords_lt=spots[i]['p']['l']
        p_coords_rt=spots[i]['p']['r']
        c_coords_lt=spots[i]['c']['l']
        c_coords_rt=spots[i]['c']['r']
        for j in range(len(p_coords_lt)):
            print(p_coords_rt[j],p_coords_lt[j])      
            depths[i]['p'].append(tri.find_depth(p_coords_rt[j],p_coords_lt[j],frame_right,frame_left,B,f,alpha))
        for j in range(len(c_coords_lt)):         
            print(c_coords_rt[j],c_coords_lt[j])   
            depths[i]['c'].append(tri.find_depth(c_coords_rt[j],c_coords_lt[j],frame_right,frame_left,B,f,alpha))

    print(depths)
    draw_depths(spots,depths,frame_left,frame_right)

    # Display the frames with drawings
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(frame_left,cv2.COLOR_BGR2RGB))
    axes[0].set_title("Left Image")
    axes[1].imshow(cv2.cvtColor(frame_right,cv2.COLOR_BGR2RGB))
    axes[1].set_title("Right Image")
    plt.show()

    # Assuming we have our depth values calculated properly:
    """
    circles_right = (x, y)
    circles_left = (x, y)
    depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left, B, f, alpha)

    cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    cv2.putText(frame_right, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    cv2.putText(frame_left, "Distance: " + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0),2)
    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    print("Depth: ", depth)                                            


    # Show the frames
    # cv2.imshow("frame right", frame_right) 
    # cv2.imshow("frame left", frame_left)
    # cv2.imshow("mask right", mask_right) 
    # cv2.imshow("mask left", mask_left)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(frame_left)
    axes[0].set_title("Left Image")
    axes[1].imshow(frame_right)
    axes[1].set_title("Right Image")
    plt.show()



    # Release and destroy all windows before termination
    # cap_right.release()
    # cap_left.release()

    # cv2.destroyAllWindows()
    """

def obj_det(frame, model):
    
    results = model(frame) 

    detections = results[0].boxes
    # print("dets", detections)
    # print('dets type', type(detections))

    return detections

def get_centers(detections):
    """
    ATTENTION: POTENTIAL ERROR
    If detected objects between the two frames are not the same, the function will not work properly given the probability that one of the objects in any given frame could go by unnoticed by the model(YOLO) given the model's accuracy.

    Even if we assume that the model is 100% accurate, the function will not work properly if the order of the detected objects is not the same in both frames. In order to calculate the depth of a given car, we need to know the corresponding car in the other frame. If the order of the detected objects is different, we will not be able to match them correctly.

    In other words, the tri.find_depth function takes circles_right and circles_left as arguments. How do I know that the circles_right and circles_left belong to the same object? 

    

    EXAMPLE:
    one image has 1 person and 3 cars, the other image has 1 person and 5 cars. The function will not work properly because the number of detected objects is different in each frame since we're tracking objects that coexist in both frames. Suggestion: dump the objects that are not in both frames. Well, cute, but how do we know which object is which?

    Example output:
    0: 640x480 1 person, 3 cars, 153.5ms
    Speed: 4.0ms preprocess, 153.5ms inference, 1.0ms postprocess per image at shape (1, 3, 640, 480)

    0: 640x480 1 person, 5 cars, 131.3ms
    Speed: 3.0ms preprocess, 131.3ms inference, 2.0ms postprocess per image at shape (1, 3, 640, 480)
    left frame:
    cars [(67, 1779), (363, 1815), (1435, 1964), (949, 1972), (577, 1859)]
    humans [(1388, 1812)]

    right frame:
    cars [(280, 1806), (1250, 1994), (1506, 1989)]
    humans [(874, 1855)]
    """

    cars = []
    humans = []

    # Filter detections for cars and humans RIGHT FRAME 
    for box in detections:
        class_id = int(box.cls[0])
        if class_id == 2:  # Car
            cars.append((box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
        elif class_id == 0:  # Human
            humans.append((box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))


    # getting the centers of the detected objects
    cars_centers = []
    humans_centers = []

    for car in cars:
        x1, y1, x2, y2 = car
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cars_centers.append((center_x, center_y))

    for human in humans:
        x1, y1, x2, y2 = human
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        humans_centers.append((center_x, center_y))

    return cars_centers, humans_centers
        
def pixel_to_3d(u, v, depth, K):
    """
    Converts a pixel coordinate (u, v) and its depth value to a 3D world coordinate.
    
    Args:
        u, v: Pixel coordinates.
        depth: Depth value from the depth map.
        K: Camera intrinsic matrix.
    
    Returns:
        (X, Y, Z) in meters.
    """
    # print("DEPTH:", depth)
    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal points

    # print('fx, fy: ', fx, fy)
    # print('cx, cy: ', cx, cy)
    
    # Convert depth (inverting if necessary)
    # if isinstance(depth, torch.Tensor):
    #     depth = depth.cpu()
    """ATTENTION: HARDCODED PARAMETER"""
    # depth = depth / 100
    depth_in_meters = True
    if depth_in_meters:
        Z = depth
    else:
        Z = 1.0 / (depth + 1e-6)  # Convert inverse depth to real-world depth

    # Convert pixel coordinates to real-world coordinates
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # X = X.cpu().numpy()
    # Y = Y.cpu().numpy()
    # print("X, Y, Z", X, Y, Z)

    # X = X.cpu()
    # Y = Y.cpu()

    # print("X, Y, Z", X, Y, Z)
    return np.array([X, Y, Z])

def compute_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def closest_centers_to_roi(x_min,x_max,centers):
    roi_center=((x_min+x_max)//2,1500)
    in_roi=[obj for obj in centers if obj[0] > x_min and obj[0]<x_max]
    return in_roi
    # return sorted(in_roi,key=lambda p:euc(p,roi_center))

def draw_spot_coordinates(spots, frame_left, frame_right):
    """
    Draws all person and car coordinates from left and right images onto the given frames.

    Parameters:
        spots (list): List of dictionaries containing 'p' (person) and 'c' (car) with 'l' and 'r' coordinates.
        frame_left (numpy.ndarray): The left image/frame.
        frame_right (numpy.ndarray): The right image/frame.

    Returns:
        tuple: Updated (frame_left, frame_right) with drawn coordinates.
    """
    # Define colors for drawing
    person_color = (255, 0, 0)  # Green
    car_color = (255, 0, 0)     # Blue
    radius = 100
    thickness = 2  # Filled circle

    for spot in spots:
        # Draw person coordinates
        for coord in spot['p']['l']:
            cv2.circle(frame_left, coord, radius, person_color, thickness)
        for coord in spot['p']['r']:
            cv2.circle(frame_right, coord, radius, person_color, thickness)
        
        # Draw car coordinates
        for coord in spot['c']['l']:
            cv2.circle(frame_left, coord, radius, car_color, thickness)
        for coord in spot['c']['r']:
            cv2.circle(frame_right, coord, radius, car_color, thickness)

    return frame_left, frame_right

def draw_depths(spots,depths, frame_left, frame_right):
    color=(0,0,0)

    for i in range(len(spots)):
        # Draw person coordinates
        for j in range(len(spots[i]['p']['l'])):
            cv2.putText(frame_left,str(depths[i]['p'][j]),spots[i]['p']['l'][j],cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
        for j in range(len(spots[i]['p']['r'])):
            cv2.putText(frame_right,str(depths[i]['p'][j]),spots[i]['p']['r'][j],cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
        
        # Draw car coordinates
        for j in range(len(spots[i]['c']['l'])):
            cv2.putText(frame_left,str(depths[i]['c'][j]),spots[i]['c']['l'][j],cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
        for j in range(len(spots[i]['c']['r'])):
            cv2.putText(frame_right,str(depths[i]['c'][j]),spots[i]['c']['r'][j],cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)



"""AUF DER HEIDE BLÜHT EIN KLEINES BLUMLEIN UND DAS HEIßT ERIKA"""
if __name__ == "__main__":
    demonstrate()
    # obsoleteMain()
