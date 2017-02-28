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
# from scipy.interpolate import UnivariateSpline, CubicSpline
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# import scipy.interpolate
from sklearn import linear_model
import math


IMG_CUTOFF = 320

class LineModel():
    """ Represents a linear hypothesis for a single lane. 
        Attributes:
            offset (float): shortest (perpendicular) distance to the lane in meters
            orientation (float): that the lane is offset from dead-ahead (+ slope means + degrees)
    """

    def __init__(self, m, b, height, width, widthInMeters=3.0):
        center = width / 2.0
        nose_height = IMG_CUTOFF
        pixel_offset = LineModel.perpendicularDistancePixels(x0=center, y0=nose_height, slope=m, intercept=b)
        self.offset = LineModel.pixelsToMeters(pixel_offset, pixel_width=width, meters_width=widthInMeters)
        self.orientation = - math.degrees(math.atan(m)) - 90

    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        return pixel_offset * meters_width / pixel_width


# def selectRandomPoints(x_all, y_all):
#     bottom_indices = np.where(y_all == IMG_CUTOFF)[0]
#     print(bottom_indices.shape)

#     if np.nonzero(bottom_indices)[0].size == 0:
#         return None, None

#     while True:
#         current_bottom = np.random.choice(bottom_indices, size=1)
#         bottom_x = x_all[current_bottom]
#         bottom_y = y_all[current_bottom]
#         print(bottom_x, bottom_y)
#         indices = np.random.randint(low=0, high=x_all.size - 1, size=2)
#         xnew = x_all[indices]
#         ynew = y_all[indices]
#         print(xnew, ynew)
#         x = np.concatenate((bottom_x, xnew))
#         y = np.concatenate((bottom_y, ynew))
#         print(y[0], y[1], y[2])
#         if y[0] < y[1] < y[2]:
#             break;
#     return x, y


# def plotSpline(img):
#     # return
#     print('PlotSpline')
#     # x = np.array([ 2.,  1.,  1.,  2.,  2.,  4.,  4.,  3.])
#     # y = np.array([ 1.,  2.,  3.,  4.,  2.,  3.,  2.,  1.])

#     gray = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), flipCode=0) # flip over x-axis
#     y_all, x_all = np.nonzero(gray)

#     # cv2.imshow('grey', gray)
#     # cv2.waitKey(3000)

#     # for xcoord, ycoord in zip(x, y):
#     #     cv2.circle(img, (xcoord, ycoord), radius=3, color=(0,255,0))
#     # print(x.size)
#     if x_all.size < 50:
#         print('\tWARNING: Not enough points to fit curve')
#         return img
#     plt.axis((0,540,0,540))
    
#     # plt.show()
#     # return

#     x, y = selectRandomPoints(x_all, y_all)
#     if x is None:
#         print('No viable points found.')
#         return
#     plt.scatter(x, y, s=3, marker='o', label='poly')

#     t = np.arange(x.shape[0], dtype=float)
#     t /= t[-1]
#     nt = np.linspace(0, 1, 100)
#     x1 = scipy.interpolate.spline(t, x, nt)
#     y1 = scipy.interpolate.spline(t, y, nt)
#     plt.plot(x1, y1, label='range_spline')

#     t = np.zeros(x.shape)
#     t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
#     t = np.cumsum(t)
#     t /= t[-1]
#     x2 = scipy.interpolate.spline(t, x, nt)
#     y2 = scipy.interpolate.spline(t, y, nt)
#     plt.plot(x2, y2, label='dist_spline')

#     plt.legend(loc='best')
#     plt.show()


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


def getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight):
    # pts1 = np.float32([[382, 48], [411, 48], [292, 565], [565, 565]])
    # pts2 = np.float32([[0,0],[100,0],[0,1600],[100,1600]])
    pts1 = np.float32([ topLeft, topRight, bottomLeft, bottomRight ])
    pts2 = np.float32([[0,0], [540,0], [0,540], [540,540]])   
    M = cv2.getPerspectiveTransform(pts1,pts2)  
    return M


def addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight):
    cv2.circle(img, topLeft, radius=5, color=(0,0,255))
    cv2.circle(img, topRight, radius=5, color=(0,0,255))
    cv2.circle(img, bottomLeft, radius=5, color=(0,0,255))
    cv2.circle(img, bottomRight, radius=5, color=(0,0,255))


def extractColor(img):
    # green = np.uint8([[[0,255,0 ]]])
    # hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
    # print hsv_green # [[[ 60 255 255]]]
    # yellow: cvScalar(20, 100, 100), cvScalar(30, 255, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 70, 30])
    upper_yellow = np.array([60, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # Threshold the HSV image to get only blue colors
    res = cv2.bitwise_and(img, img, mask= mask) # Bitwise-AND mask and original image
    # answer = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    return res


def extractEdges(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 255, apertureSize=5)
    bgrEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return bgrEdges


def dilateAndErode(img, n_dilations, n_erosions):
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=n_dilations)
    morphed = cv2.erode(dilated, kernel, iterations=n_erosions)
    return morphed


def skeleton(original):
    # kernel = np.ones((10,10),np.uint8)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # return closed

    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(original, kernel, iterations=0)
    morphed = cv2.erode(dilated, kernel, iterations=0)

    gray = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
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


def fitRobustLines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # line cv2.fitLine(gray, distType=cv2.CV_DIST_L2, param=0, reps, aeps[, line]) 
    # im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # # then apply fitline() function
    # [vx,vy,x,y] = cv2.fitLine(contours[0] ,cv2.DIST_L2,0,0.01,0.01)

    # # Now find two extreme points on the line to draw line
    # lefty = int((-x*vy/vx) + y)
    # righty = int(((gray.shape[1]-x)*vy/vx)+y)

    # #Finally draw the line
    # cv2.line(img,(gray.shape[1]-1,righty),(0,lefty),255,2)
    # return img

    # coords = cv2.flip(gray, flipCode=0) # flip over x-axis
    y, x = np.nonzero(gray)
    if x.size < 50:
        print('No lane detected.')
        cv2.putText(img, 'No lane detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        return img

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(x.reshape(x.size, 1), y.reshape(x.size, 1))
    m = model_ransac.estimator_.coef_
    b = model_ransac.estimator_.intercept_
    print('RANSAC guess:, y = {}x + {}'.format(m, b))
    mymodel = LineModel(m, b, height=img.shape[0], width=img.shape[1], widthInMeters=3.0)

    cv2.line(img=img, pt1=(0,b), pt2=(img.shape[1],m*img.shape[1]+b), color=(255,0,0), thickness=2)
    text = 'RANSAC guess:, y = {}x + {}'.format(m, b) #, offset {} orientation {}, mymodel.offset, mymodel.orientation)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return img


def addLines(img):
    copy = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    minLineLength = 100
    maxLineGap = 10
    # lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength,maxLineGap)
    # if lines is not None:
    #     for line in lines[:5]:
    #         x1,y1,x2,y2 = line[0]
    #         cv2.line(copy,(x1,y1),(x2,y2),(0,0,255),2)

    # print lines
    # print len(lines)
    # print len(lines[0])
    lines = cv2.HoughLines(image=gray, rho=1, theta=np.pi/180, threshold=100)
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
    cv2.line(img=copy, pt1=(0, IMG_CUTOFF), pt2=(540, IMG_CUTOFF), color=(0,255,0), thickness=2)
    return copy


def fitCurve(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y, x = np.nonzero(gray)

    # for xcoord, ycoord in zip(x, y):
    #     cv2.circle(img, (xcoord, ycoord), radius=3, color=(0,255,0))
    # print(x.size)
    if x.size < 50:
        print('\tWARNING: Not enough points to fit curve')
        return img

    # TODO x argument must be "strictly" increasing, but this prevents us from handling
    #   splines that are nearly vertical. We'll need a whole new way to fit the curve.
    
    # sortX, sortY = zip(*sorted(zip(x, y)))
    # print(sortX)

    # curve = UnivariateSpline(x=sortX, y=sortY, k=3, s=None) #CubicSpline
    # xs = np.arange(0, img.shape[1], 10)
    # ys = curve(xs)
    # for i in range(xs.size):
    #     if not np.isnan(ys[i]) and 0 < ys[i] < img.shape[0]:
    #         pt = xs[i], int(ys[i])
    #         print(ys[i])
    #         cv2.circle(img, pt, radius=5, color=(0,0,255))
    #         # pt1 = xs[i], int(ys[i])
    #         # pt2 = xs[i + 1], int(ys[i + 1])
    #         # print(pt1, pt2)
    #         # cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0,0,255), thickness=2)
    return img


def resizeFrame(img, scale):
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def openVideo():
    print('Load video...')
    prefix = '../../'
    cap = cv2.VideoCapture(prefix + 'taxi_intersect.mp4') # framerate of 29.97
    # cap = cv2.VideoCapture(prefix + 'taxi_trim.mp4') # framerate of 29.97
    # print('Frame size:', frame.shape) # 1920 x 1080 original, 960 x 540 resized
    return cap


def addLabels(per, mask, background, colored, lines):
    cv2.putText(per, 'Perspective', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(mask, 'BackgroundMotionSubtraction', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(background, 'Background', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(colored, 'Yellow+dilation+erosion', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(lines, 'Skeleton+HoughLines', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return per, mask, background, colored, lines


def show7(img, empty, per, mask, background, colored, lines):
    scale = 0.5
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    empty = cv2.resize(empty, dsize=None, fx=scale, fy=scale)
    per = cv2.resize(per, dsize=None, fx=scale, fy=scale)
    mask = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    background = cv2.resize(background, dsize=None, fx=scale, fy=scale)
    colored = cv2.resize(colored, dsize=None, fx=scale, fy=scale)
    lines = cv2.resize(lines, dsize=None, fx=scale, fy=scale)

    top = np.hstack((img, per, background))
    bottom = np.hstack((empty, mask, colored, lines))
    cv2.imshow('combined', np.vstack((top, bottom)))


def laneDetection(frame, fgbg, perspectiveMatrix, scaled_height, highres_scale):
    img = resizeFrame(frame, highres_scale)
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspective = cv2.warpPerspective(img, perspectiveMatrix, (scaled_height,scaled_height) )
    fgmask = fgbg.apply(perspective, learningRate=0.5)
    background = fgbg.getBackgroundImage()
    
    colored = extractColor(background)
    # edges = extractEdges(background)
    dilatedEroded = dilateAndErode(colored, n_dilations=2, n_erosions=4)
    skeletoned = skeleton(dilatedEroded)
    curve = fitRobustLines(skeletoned)
    # curve = addLines(skeletoned)
    # curve = fitCurve(skeletoned)
    # plotSpline(skeletoned)
    addPerspectivePoints(img, topLeft, topRight, bottomLeft, bottomRight)
    per, mask, back, col, lin = addLabels(  perspective, 
                                            cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), 
                                            background, 
                                            colored, 
                                            curve)
    show7(img, np.zeros((img.shape[0], img.shape[1]-background.shape[1], 3), np.uint8), per, mask, back, col, lin)


def pictureDemo(path, highres_scale=0.5, scaled_height=540):
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspectiveMatrix = getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    prefix = '../img/taxi/'
    frame = cv2.imread(prefix + path)
    frame = resizeFrame(frame, 0.5)
    img = resizeFrame(frame, highres_scale)
    laneDetection(img, fgbg, perspectiveMatrix, scaled_height, highres_scale)
    cv2.waitKey(3000)


def videoDemo(highres_scale=0.5, scaled_height=540):
    topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    perspectiveMatrix = getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    cap = openVideo()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        laneDetection(frame, fgbg, perspectiveMatrix, scaled_height, highres_scale)
        if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # pictureDemo('taxi_straight.png')
    # pictureDemo('taxi_side.png')
    # pictureDemo('taxi_curve.png')
    videoDemo()
