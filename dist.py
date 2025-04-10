import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
from ultralytics import YOLO


# Functions
# import HSV_filter as hsv
# import shape_recognition as shape
import triangulation as tri
#import calibration as calib


# Open both cameras
# cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)                    
# cap_left =  cv2.VideoCapture(1, cv2.CAP_DSHOW)
def main():
    frame_right = cv2.imread('samples/rt/rt3.jpg')
    frame_left = cv2.imread('samples/lt/lt3.jpg')

    frame_rate = 120    #Camera frame rate (maximum at 120 fps)

    B = 75             #Distance between the cameras [cm]
    f = 26.7             #Camera lense's focal length [mm]
    alpha = 68.0       #Camera field of view in the horisontal plane [degrees]

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


    
    #conf scores of all detections
    confs_right=[box.conf[0].item() for box in detections_right]
    confs_left=[box.conf[0].item() for box in detections_left]

    # print('left frame: \n', 'cars', car_centers_left, '\n', 'humans', human_centers_left, '\n', 'confs', confs_left, '\n')
    # print('right frame: \n', 'cars', car_centers_right, '\n', 'humans', human_centers_right, '\n', 'confs', confs_right, '\n')   

    
    # # Drawing the centers on frames
    # temp_r= car_centers_right + human_centers_right
    # temp_l= car_centers_left + human_centers_left

    # for (x, y) in temp_r:
    #     cv2.circle(frame_right, (x, y), radius=10, color=(0, 255, 0), thickness=2)

    # for (x, y) in temp_l:
    #     cv2.circle(frame_left, (x, y), radius=5, color=(0, 255, 0), thickness=2)

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
    # car_depths = tri.find_depth(car_centers_right, car_centers_left, frame_right, frame_left, B, f, alpha)
    # human_depths = tri.find_depth(human_centers_right, human_centers_left, frame_right, frame_left, B, f, alpha)
    # print("Car depths: ", car_depths)
    # print("Human depths: ", human_depths)

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




#Initial values

# ret_right, frame_right = cap_right.read()
# ret_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

#frame_right, frame_left = calib.undistorted(frame_right, frame_left)

########################################################################################

# If cannot catch any frame, break

# APPLYING HSV-FILTER:
# mask_right = hsv.add_HSV_filter(frame_right, 1)
# mask_left = hsv.add_HSV_filter(frame_left, 0)

# Result-frames after applying HSV-filter mask
# res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
# res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left) 

# APPLYING SHAPE RECOGNITION:
# circles_right = shape.find_circles(frame_right, mask_right)
# circles_left  = shape.find_circles(frame_left, mask_left)


# Hough Transforms can be used aswell or some neural network to do object detection

def obj_det(frame):
    # Load YOLOv5 model
    model = YOLO("yolo11s.pt")
    
    results = model(frame) 

    detections = results[0].boxes

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
    # Camera intrinsics
    fx, fy = K[0, 0], K[1, 1]  # Focal lengths
    cx, cy = K[0, 2], K[1, 2]  # Principal point

    # Convert depth (inverting if necessary)
    # if isinstance(depth, torch.Tensor):
    #     depth = depth.cpu().numpy()
    Z = 1.0 / (depth + 1e-6)  # Convert inverse depth to real-world depth

    # Convert pixel coordinates to real-world coordinates
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    # print("X, Y, Z", type(X), type(Y), type(Z))
    return np.array([X, Y, Z])


def compute_distance(p1, p2):
    """
    Computes the Euclidean distance between two 3D points.
    
    Args:
        p1, p2: Two 3D points (X, Y, Z).
    
    Returns:
        Euclidean distance in meters.
    """
    return np.linalg.norm(p1 - p2)

def closest_centers_to_roi(x_min,x_max,centers):
    roi_center=((x_min+x_max)//2,1500)
    in_roi=[obj for obj in centers if obj[0] > x_min and obj[0]<x_max]
    return in_roi
    # return sorted(in_roi,key=lambda p:euc(p,roi_center))

def euc(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

import cv2

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


main()
