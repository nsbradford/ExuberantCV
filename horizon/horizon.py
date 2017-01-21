""" 
    horizon.py
    Nicholas S. Bradford

    Use algorithm from "Vision-guided flight stability and control for micro
        air vehicles" (Ettinger et al. ).
    Intuition: horizon will be a line dividing image into two segments with low variance,
        which can be modeled as minimizing the product of the three eigenvalues of the
        covariance matrix (the determinant). 

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_line_mask(rows, columns, m, b):
    """ Params:
            rows
            columns
            m
            b
        Returns:
            rows x columns np.array boolean mask with True for all values above the line
    """
    mask = np.zeros((rows, columns), dtype=np.bool)
    for y in range(rows):
        for x in range(columns):
            if y > m * x + b:
                mask[y, x] = True
    return mask


def split_img_by_line(img, m, b):
    """ Params:
            m: slope
            b: y-intercept
        Returns:
            (arr1, arr2): two np.arrays with 3 columns (RGB) and N rows (one for each pixel)
    """
    mask = img_line_mask(rows=img.shape[0], columns=img.shape[1], m=m, b=b)
    # mask = img_line_mask(10,10,m,b)
    assert len(mask.shape) == 2
    assert mask.shape[0] == img.shape[0]
    assert mask.shape[1] == mask.shape[1]
    segment1 = img[mask]
    segment2 = img[np.logical_not(mask)]
    reshape1 = segment1.reshape(-1, segment1.shape[-1])
    reshape2 = segment2.reshape(-1, segment2.shape[-1])
    assert reshape1.shape[1] == reshape2.shape[1] == 3
    return (reshape1, reshape2)


def main():
    print ('load img...')
    img = cv2.imread('../img/ocean.jpg') #'../img/runway1.JPG' taxi_empty.jpg
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    seg1, seg2 = split_img_by_line(img, m=0.0, b=90)
    print(np.cov(seg1), np.cov(seg2))


if __name__ == '__main__':
    main()