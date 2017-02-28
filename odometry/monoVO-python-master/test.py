"""
    Source: https://github.com/uoip/monoVO-python
"""
import numpy as np 
import cv2
import math

from visual_odometry import PinholeCamera, VisualOdometry

def test():
    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam, '/home/xxx/datasets/KITTI_odometry_poses/00.txt')

    traj = np.zeros((600,600,3), dtype=np.uint8)

    for img_id in xrange(4541):
        img = cv2.imread('/home/xxx/datasets/KITTI_odometry_gray/00/image_0/'+str(img_id).zfill(6)+'.png', 0)

        vo.update(img, img_id)

        cur_t = vo.cur_t
        if(img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x)+290, int(z)+90
        true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

        cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
        cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

    cv2.imwrite('map.png', traj)


def openVideo():
    print('Load video...')
    prefix = '../../../'
    # cap = cv2.VideoCapture(prefix + 'taxi_intersect.mp4') # framerate of 29.97
    cap = cv2.VideoCapture(prefix + 'taxi_trim.mp4') # framerate of 29.97
    # print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
    return cap


def resizeFrame(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


# rotation matrix -> euler angle from 
#       https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([math.degrees(angle) for angle in [x, y, z]])


def vidTest():
    # WARNING - this camera calibration is wrong
    # fx=718.8560, fy=718.8560
    # focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180)
    # http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/
    FOV = 35
    focal_pixel = (540 * 0.5) / math.tan(FOV * 0.5 * math.pi / 180)
    cam = PinholeCamera(width=960, height=540, fx=focal_pixel, fy=focal_pixel, cx=960/2.0, cy=540/2.0)
    vo = VisualOdometry(cam, annotations=None) # TODO hope we don't need annotations
    traj = np.zeros((600,600,3), dtype=np.uint8)

    img_id = 0

    cap = openVideo()
    while(cap.isOpened()):
        ret, img = cap.read()
        img = resizeFrame(img, scale=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vo.update(gray, img_id)

        cur_t = vo.cur_t
        if(img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x)+290, int(z)+90
        # true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

        cv2.circle(traj, (draw_x,draw_y), 1, (img_id*255/4540,255-img_id*255/4540,0), 1)
        # cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imshow('Road facing camera', gray)
        cv2.imshow('Trajectory', traj)

        if img_id > 2: print(rotationMatrixToEulerAngles(vo.cur_R))

        img_id += 1

        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break


if __name__ == '__main__':
    vidTest()