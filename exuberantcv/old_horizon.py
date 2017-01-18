"""
    horizon.py
    Nicholas Bradford
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# polar to cartesian
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def plot(img1, img2):
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    plt.show()

def horizon(img):
    """ First algo:
            1) Canny edge detector
            2) Linear regression
        Intuition: Hough lines is too sensitive to noise on the horizon

    """
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    plot(img, edges)

def horizonThreshold(color_img, grey_img):
    # thresh = cv2.adaptiveThreshold(img, 355, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #               cv2.THRESH_BINARY, blockSize=3, C=2) 
    # plot(th3, th3)

    blur = cv2.GaussianBlur(grey_img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lines = cv2.HoughLines(image=th3, rho=1, theta=np.pi/180, threshold=300)[:50]
    print (len(lines))
    # print lines
    # print len(lines)
    # print len(lines[0])
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
        cv2.line(img=color_img, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=1)
    print(color_img.shape, grey_img.shape, th3.shape)

    plt.subplot(131)
    plt.imshow(color_img)
    plt.subplot(132)
    plt.imshow(th3, cmap='gray')
    plt.subplot(133)
    plt.hist(grey_img.ravel(), 256)
    plt.show()


    # TODO otsu's method
    # http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html


if __name__ == '__main__':
    color_img = cv2.imread('../img/taxi_empty.jpg')
    grey_img = cv2.imread('../img/taxi_empty.jpg', 0) #, cv2.CV_8UC1
    # horizon(img)
    horizonThreshold(color_img, grey_img)