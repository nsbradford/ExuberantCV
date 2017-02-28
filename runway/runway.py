"""
	runway.py
	Nicholas S. Bradford
	02-2017

	Algorithm:
		1) Extract white only to get runway lane markings
		2) Use Gaussian filter to get vertical edges
		3) Fit Hough lines to lanes and center line
		4) Extract offset from centerline
		5) Extract runway number (and make sure it doesn't say "TAXI"!)

"""


def pictureDemo(path, highres_scale=0.5, scaled_height=540):
    # topLeft, topRight, bottomLeft, bottomRight = getPerspectivePoints(highres_scale)
    # perspectiveMatrix = getPerspectiveMatrix(topLeft, topRight, bottomLeft, bottomRight)
    # fgbg = cv2.createBackgroundSubtractorMOG2()
    prefix = '../img/'
    frame = cv2.imread(prefix + path)
    frame = resizeFrame(frame, 0.5)
    img = resizeFrame(frame, highres_scale)
    runwayDetection(img)
    cv2.waitKey(3000)


if __name__ == '__main__':
    pictureDemo('runway1.JPG')

