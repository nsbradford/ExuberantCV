"""
    runner.py

"""

import cv2

from lanes import getPerspectiveMatrix, laneDetection, Constants


def openVideo(filename):
    """ 1920 x 1080 original, 960 x 540 resized """ 
    print('Load video...')
    cap = cv2.VideoCapture('../vid/' + filename)
    # print('Frame size:', frame.shape)
    return cap


def resizeFrame(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def pictureDemo(path, highres_scale=0.5, scaled_height=Constants.IMG_SCALED_HEIGHT):
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    prefix = '../img/taxi/'
    frame = cv2.imread(prefix + path)
    frame = resizeFrame(frame, 0.5)
    img = resizeFrame(frame, highres_scale)
    laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale)
    cv2.waitKey(0)


def videoDemo(filename, highres_scale=0.5, scaled_height=Constants.IMG_SCALED_HEIGHT):
    perspectiveMatrix = getPerspectiveMatrix(highres_scale)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = openVideo(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        img = resizeFrame(frame, highres_scale)
        laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale)
        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pictureDemo('taxi_straight.png')
    # pictureDemo('taxi_side.png')
    pictureDemo('taxi_curve.png')
    videoDemo('taxi_intersect.mp4') # framerate of 29.97
    # videoDemo('taxi_trim.mp4') # framerate of 29.97