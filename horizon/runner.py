
import cv2 # for reading photos and videos
import numpy as np
from horizon import optimize_global, optimize_local
import plotter


def load_img(path):
    print ('load img...') # taxi_rotate.png
    img = cv2.imread(path)
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape)
    return resized


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


def main():
    #'taxi_rotate.png runway1.JPG taxi_empty.jpg ocean sunset grass
    img = load_img('../img/taxi_rotate.png') 
    print('Optimize scores...')
    answer, scores, grid = optimize_global(img)
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


def add_line_to_frame(img, m, b):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pt1 = (0, b.astype(np.int64))
    pt2 = img.shape[1] - 1, (m * (img.shape[1] - 1) + b).astype(np.int64)
    cv2.line(img=rgb_img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)
    return rgb_img


def video_demo():
    print('Load video...')
    cap = cv2.VideoCapture('../../flying_turn.avi')
    count = 0
    while(cap.isOpened()):
        # print('Read new frame...')
        ret, frame = cap.read()
        count += 1
        if count % 5 != 0:
            continue
        if not ret:
            break
        img = cv2.resize(frame, dsize=None, fx=0.1, fy=0.1)
        print('Optimize...', img.shape)
        if count < 20:
            answer, scores, grid = optimize_global(img)
        else:
            answer, scores, grid = optimize_local(img, m, b)
        m = answer[0]
        b = answer[1]
        label = 'Prediction m: ' + str(m) + ' b: ' + str(b)
        prediction = add_line_to_frame(frame, m, b*10.0)#m=np.float32(0.0), b=np.float32(30.0))
        # cv2.imshow('frame',frame)
        cv2.imshow('label', prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # time_score()2
    # main()
    video_demo()