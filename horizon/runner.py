
import numpy as np
from horizon import load_img, optimize_scores
import plotter

def main():
    #'taxi_rotate.png runway1.JPG taxi_empty.jpg ocean sunset grass
    img = load_img('../img/taxi_rotate.png') 
    print('Optimize scores...')
    answer, scores, grid = optimize_scores(img)
    # print('Max:', max_index)

    m = answer[0]
    b = answer[1]
    print('m:', m, '  b:', b)
    plotter.plot2D(img, scores, m, b)

    X = [option[0] for option in grid] # m
    Y = [option[1] for option in grid] # b
    Z = list(map(lambda x: x * (10**8), scores))
    print(len(X), len(Y), len(Z))
    plotter.scatter3D(X, Y, Z) # scatter3D(X[::10], Y[::10], Z[::10])


def basic_test():
    img = load_img('../img/taxi_rotate.png') #'../img/runway1.JPG' taxi_empty.jpg ocean sunset grass
    good_line = score_line(img, m=0.0, b=20)
    bad_line = score_line(img, m=2.0, b=0)
    assert good_line > bad_line
    print('Basic test of scoring...')


def time_score():
    import timeit
    result = timeit.timeit('horizon.score_line(img, m=0.0,  b=20)', 
                        setup='import horizon; img=horizon.load_img();', 
                        number=1000)
    print('Timing:', result/1000, 'seconds to score a single line.')


if __name__ == '__main__':
    main()