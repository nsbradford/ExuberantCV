"""
    lanes.py
    Nicholas S. Bradford
    12 Feb 2017

"""

import cv2 # for reading photos and videos
import numpy as np
from showmany import multi_plot

def getPerspectiveMatrix(scaled_width, scaled_height):
    # pts1 = np.float32([[382, 48], [411, 48], [292, 565], [565, 565]])
    # pts2 = np.float32([[0,0],[100,0],[0,1600],[100,1600]])   
    horizon_height = int(scaled_height / 3.0)
    pts1 = np.float32([ [scaled_width / 2.0 - 1, horizon_height], 
                        [scaled_width / 2.0 - 1, horizon_height], 
                        [0, scaled_height], 
                        [scaled_width, scaled_height]])
    pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])   
    M = cv2.getPerspectiveTransform(pts1,pts2)  
    return M


def video_demo(highres_scale=0.5):
    original_width = 1920
    original_height = 1080
    scaled_width = int(highres_scale * original_width)
    scaled_height = int(highres_scale * original_height)
    horizon_height = int(scaled_height / 3.0)
    wing_height = int(scaled_height * 2.0 / 3.0)
    perspectiveMatrix = getPerspectiveMatrix(scaled_width, scaled_height)

    print('Load video...')
    prefix = '../../'
    cap = cv2.VideoCapture(prefix + 'TaxiOut1.mp4') # framerate of 29.97
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break;
        img = cv2.resize(frame, dsize=None, fx=1.0 * highres_scale, fy=1.0 * highres_scale)
        print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
        # perspective = cv2.warpPerspective(img, perspectiveMatrix, (300,300) )   

        cv2.circle(img, (int(scaled_width / 2.0 - 1), horizon_height), radius=5, color=(0,0,255))
        cv2.circle(img, (int(scaled_width / 2.0 + 1), horizon_height), radius=5, color=(0,0,255))
        cv2.circle(img, (scaled_width, wing_height), radius=5, color=(0,0,255))
        cv2.circle(img, (scaled_width, wing_height), radius=5, color=(0,0,255))

        cv2.imshow('resized', img)
        # cv2.imshow('perspective', perspective)
        # multi_plot("Images", 2, img, perspective);

        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # time_score()2
    # main()
    video_demo()