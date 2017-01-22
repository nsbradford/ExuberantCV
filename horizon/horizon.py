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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def img_line_mask(rows, columns, m, b):
    """ Params:
            rows
            columns
            m
            b
        Returns:
            rows x columns np.array boolean mask with True for all values above the line
    """
    # TODO: there must be a way to optimize this
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
    # print('\tLine:', m, b)
    mask = img_line_mask(rows=img.shape[0], columns=img.shape[1], m=m, b=b)
    assert len(mask.shape) == 2
    assert mask.shape[0] == img.shape[0]
    assert mask.shape[1] == mask.shape[1]
    segment1 = img[mask]
    segment2 = img[np.logical_not(mask)]
    reshape1 = segment1.reshape(-1, segment1.shape[-1])
    reshape2 = segment2.reshape(-1, segment2.shape[-1])
    assert reshape1.shape[1] == reshape2.shape[1] == 3
    return (reshape1, reshape2)


def compute_variance_score(segment1, segment2):
    """ Params:
            segment1 (np.array): n x 3, where n is number of pixels in first segment
            segment2 (np.array): n x 3, where n is number of pixels in first segment
        Returns:
            F (np.double): the score for these two segments (higher = better line hypothesis)
    """
    # print('Covariance matrices: )
    # print(np.cov(seg1), np.cov(seg2))
     # linalg.eigh() is more stable than np.linalg.eig, but only for symmetric matrices
    assert segment1.shape[1] == segment2.shape[1] == 3
    # TODO shouldn't be wasting time on impossible hypotheses
    if segment1.shape[0] < 2 or segment2.shape[0] < 2:
        raise RuntimeError('Invalid hypothesis.')
    cov1 = np.cov(segment1.T)
    cov2 = np.cov(segment2.T)
    assert cov1.shape == cov2.shape == (3,3)
    # print('Covariance matrices:', cov1, cov2)
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
    # print('Score', img.shape, m, b)
    seg1, seg2 = split_img_by_line(img, m=m, b=b)
    # print('\tSegment shapes: ', seg1.shape, seg2.shape)
    score = compute_variance_score(seg1, seg2)
    return score


def load_img(path):
    print ('load img...') # taxi_rotate.png
    img = cv2.imread(path)
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape)
    return resized


def optimize_scores(img):
    pass
    # convert (pitch angle, bank angle) to (slope, intercept)
    # grid = []
    # pitch_range = 1.0 # TODO
    # bank_range = 1.0 # TOOD
    # for i in range(pitch_range):
    #     for j in range(bank_range):
    #         grid.append(i, j)
    # scores = list(map(lambda x: score_line(img, x[0], x[1]), grid))
    # max_index = np.argmax(scores)
    # answer = grid[max_index]
    grid = []
    for b in range( 1, img.shape[0] - 2):
        for m in range(- (img.shape[0] - 1), img.shape[1] - 1):
            grid.append((m, b))
    scores = list(map(lambda x: score_line(img, x[0], x[1]), grid))
    # for i in range(len(scores)): print(i, ':', scores[i])
    return scores, grid


def plot3D(X, Y, Z):
    print('Plot in 3D...')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter 

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # print(X.shape, Y.shape, len(Z))
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #         linewidth=0, antialiased=False)
    surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    # ax.set_zlim(-1.01, 1.01)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # print (Z)
    plt.show()

def scatter3D(X, Y, Z):
    print('Plot in 3D...')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=np.abs(Z), cmap=cm.coolwarm)

    ax.set_xlabel('M (slope)')
    ax.set_ylabel('B (intercept)')
    ax.set_zlabel('Z Label')

    plt.show()


def main():
    img = load_img('../img/taxi_rotate.png') #'../img/runway1.JPG' taxi_empty.jpg ocean sunset grass
    good_line = score_line(img, m=0.0, b=20)
    bad_line = score_line(img, m=2.0, b=0)
    assert good_line > bad_line
    print('Basic test of scoring works...')

    scores, grid = optimize_scores(img)
    max_index = np.argmax(scores)
    answer = grid[max_index]
    print('Max:', max_index)

    m = answer[0]
    b = answer[1]

    print('m:', m, '  b:', b)
    pt1 = (0, b)
    pt2 = img.shape[1] - 1, m * (img.shape[1] - 1) + b
    print (pt2)

    cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    plt.subplot(122)
    y = list(map(lambda x: x * (10**8), scores))
    x = list(range(len(scores)))
    plt.fill_between(x, y, [0 for x in range(len(scores))], color='grey')
    # plt.imshow(resized),plt.title('Output')
    plt.show()
    X = [option[0] for option in grid] # m
    Y = [option[1] for option in grid] # b
    Z = y
    print(len(X), len(Y), len(Z))
    # plot3D(X[::20], Y[::20], Z[::20])
    # scatter3D(X[::10], Y[::10], Z[::10])
    scatter3D(X, Y, Z)


def time_score():
    import timeit
    result = timeit.timeit('horizon.score_line(img, m=0.0,  b=20)', 
                        setup='import horizon; img=horizon.load_img();', 
                        number=1000)
    print('Timing:', result/1000, 'seconds to score a single line.')


if __name__ == '__main__':
    main()
    # time_score()