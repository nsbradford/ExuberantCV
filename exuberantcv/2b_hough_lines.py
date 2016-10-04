""" 
    test.py 
    Nicholas S .Bradford

    Hough Lines
    http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from plotter import plot3

def old_hough_lines():
    img = cv2.imread('../img/runway1.jpg')
    copy = img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img=copy, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)

    # plt.subplot(121),plt.imshow(copy),plt.title('Input')
    # plt.subplot(122),plt.imshow(img),plt.title('Output')
    # plt.show()
    # cv2.imwrite('houghlines3.jpg',img)

def hough_lines(img):
    copy = img.copy()
    gray = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    plt.subplot(133),plt.imshow(edges),plt.title('Lines')
    plt.show()

    lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi/180, threshold=250)
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
        cv2.line(img=copy, pt1=(x1,y1), pt2=(x2,y2), color=(255,0,0), thickness=2)
    return edges, copy

def main():
    print 'load img...'
    img = cv2.imread('../img/runway1.jpg')
    edges, copy = hough_lines(img)
    plot3(img, edges, copy)


if __name__ == '__main__':
    main()