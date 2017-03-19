"""
    odometry.py
    Nicholas S. Bradford
    19 March 2016


    Algorithm from http://avisingh599.github.io/assets/ugp2-report.pdf:

    1) Capture and undistort two consecutive images.
    2) Use FAST algorithm to detect features in I^t, and track features in I^t+1. 
        New detection is triggered if the # of features drops below a threshold.
    3) Use Nister's 5-point algorithm with RANSAC to compute essential matrix
        Benzun's advice: will work, but will always have inaccuracy.
    4) Estimate R, t from essential matrix
    6) Add R to current rotation angle estimate (Kalman filter?)
    7)

"""

import math
import numpy as np
import cv2


class Rotation():
    """
        Attributes:
            theta: angle in degrees 
    """

    def __init__(self):
        self.theta = math.pi / 2
        self.img_size = 512
        self.img_shape = (self.img_size, self.img_size, 3)
        self.center = (self.img_size//2, self.img_size//2)
    
    def update(self, angle):
        self.theta += math.radians(angle)
        self.display()

    def display(self):
        img = np.zeros(self.img_shape, np.uint8) # Create a black image
        cv2.circle(img, center=self.center, radius=50, color=(0,0,255), thickness=1)
        self.add_line(img)
        cv2.namedWindow('Display Window', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Rotation orientation', img)
        cv2.waitKey(0)

    def add_line(self, img):   
        x2 = int(self.center[0] - 1000 * np.cos(self.theta))
        y2 = int(self.center[1] - 1000 * np.sin(self.theta))
        cv2.line(img=img, pt1=self.center, pt2=(x2,y2), color=(255,255,255), thickness=2)
        

def main():
    rot = Rotation()
    rot.update(45)
    rot.update(-135)
    # '../vid/rotate.mp4'


if __name__ == '__main__':
    main()