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

# Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# class VisualOdometry:

#     def processFirstFrame(self):
#         self.px_ref = self.detector.detect(self.new_frame)
#         self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
#         # self.frame_stage = STAGE_SECOND_FRAME

#     def processSecondFrame(self):
#         self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
#         E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#         _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
#         # self.frame_stage = STAGE_DEFAULT_FRAME 
#         # self.px_ref = self.px_cur

#     def processFrame(self, frame_id):
#         self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
#         E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#         _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
#         # absolute_scale = self.getAbsoluteScale(frame_id)
#         absolute_scale = 0.5
#         if(absolute_scale > 0.1):
#             # self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
#             self.cur_R = R.dot(self.cur_R)
#         if(self.px_ref.shape[0] < kMinNumFeature):
#             self.px_cur = self.detector.detect(self.new_frame)
#             self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
#         self.px_ref = self.px_cur

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


def openVideo():
    print('Load video...')
    # cap = cv2.VideoCapture(prefix + 'taxi_intersect.mp4') # framerate of 29.97
    cap = cv2.VideoCapture('../../vid/' + 'rotate.mp4') # framerate of 29.97
    # print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
    return cap


def main():
    rot = Rotation()
    rot.update(45)
    rot.update(-135)
    # '../vid/rotate.mp4'
    rot = Rotation()
    cap = openVideo()
    while(cap.isOpened()):
        ret, img = cap.read()
        img = resizeFrame(img, scale=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break


if __name__ == '__main__':
    main()