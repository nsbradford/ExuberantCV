import cv2
import numpy as np
from plotter import plot2

def blur():
    img = cv2.imread('../img/lake.jpg')
    kernel = np.array([ [1,2,1],
                        [2,4,2],
                        [1,2,1] ], np.float32)/16
    #kernel = np.ones((5,5),np.float32)/25
    #kernel = np.asanyarray(kernel, np.float32)
    out = cv2.filter2D(img, -1, kernel) #astype(np.float32)
    plot2(img, out)

if __name__ == '__main__':
    blur()