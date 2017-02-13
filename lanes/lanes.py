"""
    lanes.py
    Nicholas S. Bradford
    12 Feb 2017

"""

import cv2 # for reading photos and videos
import numpy as np
from showmany import multi_plot



def getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight):
    # pts1 = np.float32([[382, 48], [411, 48], [292, 565], [565, 565]])
    # pts2 = np.float32([[0,0],[100,0],[0,1600],[100,1600]])
    pts1 = np.float32([ topLeft, topRight, bottomLeft, bottomRight ])
    pts2 = np.float32([[0,0], [540,0], [0,540], [540,540]])   
    M = cv2.getPerspectiveTransform(pts1,pts2)  
    return M


def extractColor(img):
    # green = np.uint8([[[0,255,0 ]]])
    # hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    # print hsv_green # [[[ 60 255 255]]]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # yellow: cvScalar(20, 100, 100), cvScalar(30, 255, 255)
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    lower_yellow = np.array([10, 100, 50])
    upper_yellow = np.array([40, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)

    answer = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return answer


def video_demo(highres_scale=0.5):
    original_width = 1920
    original_height = 1080
    scaled_width = int(highres_scale * original_width)
    scaled_height = int(highres_scale * original_height)
    horizon_height = int(scaled_height / 3.0)
    wing_height = int(scaled_height * 2.0 / 3.0)
    right_width = int(scaled_width * 2.0 / 3.0)
    left_width = int(scaled_width / 3.0)

    topLeft = (int(scaled_width / 2.0 - 50), horizon_height + 50)
    topRight = (int(scaled_width / 2.0 + 50), horizon_height + 50)
    bottomLeft = (left_width, wing_height)
    bottomRight = (right_width, wing_height)

    perspectiveMatrix = getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight)

    print('Load video...')
    prefix = '../../'
    cap = cv2.VideoCapture(prefix + 'taxi_trim.mp4') # framerate of 29.97
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break;
        img = cv2.resize(frame, dsize=None, fx=1.0 * highres_scale, fy=1.0 * highres_scale)
        # print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
        perspective = cv2.warpPerspective(img, perspectiveMatrix, (scaled_height,scaled_height) )
        colored = extractColor(perspective)

        cv2.circle(img, topLeft, radius=5, color=(0,0,255))
        cv2.circle(img, topRight, radius=5, color=(0,0,255))
        cv2.circle(img, bottomLeft, radius=5, color=(0,0,255))
        cv2.circle(img, bottomRight, radius=5, color=(0,0,255))

        # cv2.imshow('resized', img)
        # cv2.imshow('perspective', perspective)
        # multi_plot("Images", 2, img, perspective);
        combined = np.hstack((img, perspective, colored))
        cv2.imshow('combined', combined)

        if cv2.waitKey(10) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # time_score()2
    # main()
    video_demo()