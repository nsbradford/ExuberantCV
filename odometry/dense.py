"""
    dense.py
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

"""

import cv2
import numpy as np
cap = cv2.VideoCapture("../vid/taxi_intersect.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    print('.')
    ret, frame2 = cap.read()
    nextFrame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,nextFrame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    scale = 0.5
    img = cv2.resize(rgb, dsize=None, fx=scale, fy=scale)
    cv2.imshow('frame2',img)
    # k = cv2.waitKey(30) & 0xff
    if cv2.waitKey(33) & 0xFF == ord('q'): # 1000 / 29.97 = 33.37
        break
    # if k == 27:
    #     break
    # elif k == ord('s'):
    #     cv2.imwrite('opticalfb.png',frame2)
    #     cv2.imwrite('opticalhsv.png',rgb)
    prvs = nextFrame

cap.release()
cv2.destroyAllWindows()