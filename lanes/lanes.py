"""
    lanes.py
    Nicholas S. Bradford
    12 Feb 2017

    TODO:
        -RANSAC for curve/spline-fitting
        -Kalman filter (use pykalman)

"""

import cv2 # for reading photos and videos
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline


def addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight):
    cv2.circle(img, topLeft, radius=5, color=(0,0,255))
    cv2.circle(img, topRight, radius=5, color=(0,0,255))
    cv2.circle(img, bottomLeft, radius=5, color=(0,0,255))
    cv2.circle(img, bottomRight, radius=5, color=(0,0,255))

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
    lower_yellow = np.array([10, 70, 30])
    upper_yellow = np.array([60, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)

    # answer = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(res, kernel, iterations=5)
    morphed = cv2.erode(dilated, kernel, iterations=7)
    return morphed


def extractEdges(img):
    edges = cv2.Canny(img, 20, 100, apertureSize=3)
    bgrEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return bgrEdges


def skeleton(original):
    # kernel = np.ones((10,10),np.uint8)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # return closed
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    size = np.size(gray)
    skel = np.zeros(gray.shape,np.uint8)
    ret,img = cv2.threshold(gray,20,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while not done:
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    colorSkel = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    return colorSkel


def addLines(img):
    copy = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLines(image=gray, rho=1, theta=np.pi/180, threshold=100)
    # print lines
    # print len(lines)
    # print len(lines[0])
    if lines is not None:
        for line in lines: #[0]:
            #print line
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)        
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img=copy, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)
    return copy


def fitCurve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = np.nonzero(gray)
    # print(x.size)
    if x.size < 50:
        print('\tWARNING: Not enough points to fit curve')
        return img

    # TODO x argument must be "strictly" increasing, but this prevents us from handling
    #   splines that are nearly vertical. We'll need a whole new way to fit the curve.
    
    sortX, sortY = zip(*sorted(zip(x, y)))
    print(sortX)

    curve = UnivariateSpline(x=sortX, y=sortY, k=3, s=None) #CubicSpline
    xs = np.arange(0, img.shape[1], 10)
    ys = curve(xs)
    for i in range(xs.size):
        if not np.isnan(ys[i]) and 0 < ys[i] < img.shape[0]:
            pt = xs[i], int(ys[i])
            print(ys[i])
            cv2.circle(img, pt, radius=5, color=(0,0,255))
            # pt1 = xs[i], int(ys[i])
            # pt2 = xs[i + 1], int(ys[i + 1])
            # print(pt1, pt2)
            # cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0,0,255), thickness=2)
    return img


def getPerspectivePoints(highres_scale):
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
    return topLeft, topRight, bottomLeft, bottomRight


def resizeFrame(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def openVideo():
    print('Load video...')
    prefix = '../../'
    cap = cv2.VideoCapture(prefix + 'taxi_trim.mp4') # framerate of 29.97
    # print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
    return cap


def addLabels(per, mask, colored, lines):
    cv2.putText(per, 'Perspective', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(mask, 'BackgroundMotionSubtraction', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(colored, 'Yellow+dilation+erosion', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(lines, 'Skeleton+HoughLines', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return per, mask, colored, lines


def showFour(per, mask, colored, lines):
    top = np.hstack((per, mask))
    bottom = np.hstack((colored, lines))
    cv2.imshow('combined', np.vstack((top, bottom)))


def video_demo(highres_scale=0.5, scaled_height=540):
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspectiveMatrix = getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = openVideo()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break;
        img = resizeFrame(frame, highres_scale)
        perspective = cv2.warpPerspective(img, perspectiveMatrix, (scaled_height,scaled_height) )
        
        fgmask = fgbg.apply(perspective, learningRate=0.2)
        background = fgbg.getBackgroundImage()

        # edges = extractEdges(perspective)
        colored = extractColor(background)
        morphed = skeleton(colored)
        # curve = addLines(morphed)
        curve = fitCurve(morphed)
        addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight)
        per, mask, col, lin = addLabels(perspective, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), colored, curve)
        showFour(per, mask, col, lin)
        # cv2.imshow('combined', np.hstack((perspective, cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), colored, lines)))

        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_demo()
