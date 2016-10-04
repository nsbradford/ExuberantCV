""" test.py 
    Nicholas S .Bradford

    Perspective Transformation
    http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html

"""

print 'import...'
import cv2
import numpy as np
from plotter import plot2

def perspective():
    print 'load img...'
    img = cv2.imread('../img/runway1.jpg')
    rows,cols,ch = img.shape    

    # 292, 565 bottom left
    # 565, 565 bottom right
    # 382, 48 top left
    # 411, 48 top right

    print 'transform...'
    #pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts1 = np.float32([[382, 48], [411, 48], [292, 565], [565, 565]])
    pts2 = np.float32([[0,0],[100,0],[0,1600],[100,1600]])    

    M = cv2.getPerspectiveTransform(pts1,pts2)    
    dst = cv2.warpPerspective(img,M,(100,1600))    

    plot2(img, dst)


if __name__ == '__main__':
    perspective()