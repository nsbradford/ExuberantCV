""" horizon.py
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from plotter import plot3


def horizon(img):
    copy = img.copy()
    copy = cv2.GaussianBlur(img,(5,5),0)
    
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    plt.subplot(133),plt.imshow(edges),plt.title('Lines')
    # plt.show()

    lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi/180, threshold=100)
    print('Lines detected:', len(lines))
    for line in lines[:1]: #[0]:
        print('Graph line...')
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
    print ('load img...')
    img = cv2.imread('../img/taxi_empty.jpg') #'../img/runway1.JPG' taxi_empty.jpg
    edges, copy = horizon(img)
    plot3(img, edges, copy)


if __name__ == '__main__':
    main()
