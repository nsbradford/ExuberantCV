"""
    fit.py

"""

import cv2
import numpy as np
from sklearn import linear_model
import math

import lanes


class LineModel():
    """ Represents a linear hypothesis for a single lane. 
        Attributes:
            offset (float): shortest (perpendicular) distance to the lane in meters
            orientation (float): that the lane is offset from dead-ahead (+ slope means + degrees)
    """

    def __init__(self, m, b, height, width, widthInMeters=3.0):
        center = width / 2.0
        nose_height = lanes.Constants.IMG_CUTOFF
        pixel_offset = LineModel.perpendicularDistancePixels(x0=center, y0=nose_height, slope=m, intercept=b)
        self.offset = LineModel.pixelsToMeters(pixel_offset, pixel_width=width, meters_width=widthInMeters)
        raw_orientation = math.degrees(math.atan(m))
        offset = - 90 if raw_orientation >= 0 else 90
        self.orientation = raw_orientation + offset

    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        return pixel_offset * meters_width / pixel_width


def isMultiLine(x, y):
    cov = np.cov(np.vstack((x, y)))
    assert cov.shape == (2,2)
    evals, evecs = np.linalg.eigh(cov)
    ismultiline = evals[1] / 10 < evals[0]
    return ismultiline, int(evals[0]), int(evals[1])


def fitLines(img):
    """
        Args:
            img
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    roi = gray[:lanes.Constants.IMG_CUTOFF, :]
    y, x = np.nonzero(roi)
    if x.size < 50:
        print('No lane detected.')
        cv2.putText(img, 'No lane detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        return img

    # print('\t{}'.format(isMultiLine(x, y)))

    model_ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
                                                max_trials=1000)
                                                # residual_threshold=5.0 )
    model_ransac.fit(x.reshape(x.size, 1), y.reshape(x.size, 1))
    m = model_ransac.estimator_.coef_[0,0]
    b = model_ransac.estimator_.intercept_[0]
    inliers = model_ransac.inlier_mask_
    mymodel = LineModel(m, b, height=img.shape[0], width=img.shape[1], widthInMeters=3.0)

    print('\t{}/{} inliers'.format(np.count_nonzero(inliers), inliers.size))
    for i in range(inliers.size):
        if inliers[i]:
            xcoord = x[i]
            ycoord = y[i]
            cv2.circle(img, (xcoord, ycoord), radius=3, color=(0,0,255))

    # print('RANSAC:, y = {0:.2f}x + {1:.2f} offset {2:.2f} orientation {3:.2f}'.format(m, b, mymodel.offset, mymodel.orientation))
    cv2.line(img=img, pt1=(0,int(b)), pt2=(img.shape[1],int(m*img.shape[1]+b)), color=(255,0,0), thickness=2)
    cv2.line(img=img, pt1=(0, lanes.Constants.IMG_CUTOFF), pt2=(lanes.Constants.IMG_SCALED_HEIGHT, lanes.Constants.IMG_CUTOFF), 
                        color=(0,255,0), thickness=2)
    text = 'offset {0:.2f} orientation {1:.2f}'.format(mymodel.offset, mymodel.orientation)
    cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    return img







