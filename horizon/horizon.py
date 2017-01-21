""" 
    horizon.py
    Nicholas S. Bradford

    Use algorithm from "Vision-guided flight stability and control for micro
        air vehicles" (Ettinger et al. ).
    Intuition: horizon will be a line dividing image into two segments with low variance,
        which can be modeled as minimizing the product of the three eigenvalues of the
        covariance matrix (the determinant). 

    In its current form, can run 10000 iterations in 9.1 seconds, or about 30 per iteration
        at 30Hz. Java performance benefit of 10x would mean 300 per iteration,
        and moving to 10Hz would leave ~1000 per iteration.
    The initial optimized search grid is on a 12x12 grid (144 values),
        which is then refined on a full-resolution image using a gradient-descent-like
        sampling technique. (requires 4 checks at each step and ~7 steps = ~28, but
        will but at higher resolution)
    Total requirements: must be able to run at least ~200 checks/second

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
    # print('Segment shapes: ', (reshape1.shape, reshape2.shape))
    return (reshape1, reshape2)


def compute_variance_score(segment1, segment2):
    # print('Covariance matrices: )
    # print(np.cov(seg1), np.cov(seg2))
     # linalg.eigh() is more stable than np.linalg.eig, but only for symmetric matrices
    assert segment1.shape[1] == segment2.shape[1] == 3
    assert segment1.shape[0] > 3
    assert segment2.shape[0] > 3
    cov1 = np.cov(segment1.T)
    cov2 = np.cov(segment2.T)
    assert cov1.shape == cov2.shape == (3,3)
    evals1, evecs1 = np.linalg.eig(cov1)
    evals2, evecs2 = np.linalg.eig(cov2)

    # When the covariance matrix is nearly singular (due to color issues), the determinant
    # will also be driven to zero. Thus, we introduce additional terms to supplement the
    # score when this case occurs (the determinant dominates it in the normal case):
    #   where g=GROUND and s=SKY (covariance matrices) 
    #   F = [det(G) + det(S) + (eigG1 + eigG1 + eigG1)^2 + (eigS1 + eigS1 + eigS1)^2]^-1
    F = np.linalg.det(cov1) + np.linalg.det(cov2) + (np.sum(evals1) ** 2) + (np.sum(evals2) ** 2)
    return F ** -1


def score_line(img, m, b):
    seg1, seg2 = split_img_by_line(img, m=m, b=b)
    score = compute_variance_score(seg1, seg2)
    return score


def load_img():
    print ('load img...')
    img = cv2.imread('../img/ocean.jpg') #'../img/runway1.JPG' taxi_empty.jpg
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape)
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(resized),plt.title('Output')
    # plt.show()
    return resized


def main():
    img = load_img()
    good_line = score_line(img, m=0.0, b=20)
    bad_line = score_line(img, m=2.0, b=0)
    assert good_line > bad_line
    print('Basic test of scoring works.')


def time_score():
    import timeit
    result = timeit.timeit('horizon.score_line(img, m=0.0,  b=20)', 
                        setup='import horizon; img=horizon.load_img();', 
                        number=1000)
    print('Timing:', result/1000, 'seconds to score a single line.')


if __name__ == '__main__':
    main()
    # time_score()