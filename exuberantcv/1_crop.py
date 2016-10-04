import cv2
from plotter import plot2

def crop():
    print 'load img...'
    img = cv2.imread('../img/runway1.jpg')
    rows,cols,ch = img.shape
    crop = img[rows-100:rows, 100:cols-100]
    plot2(img, crop)

if __name__ == '__main__':
    crop()
