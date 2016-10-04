
import matplotlib.pyplot as plt

def plot2(original, changed):
    print 'plot...'
    plt.subplot(121),plt.imshow(original),plt.title('Input')
    plt.subplot(122),plt.imshow(changed),plt.title('Output')
    plt.show()
    print 'done.'

def plot3(img1, img2, img3):
    print 'plot...'
    plt.subplot(131),plt.imshow(img1),plt.title('1')
    plt.subplot(132),plt.imshow(img2),plt.title('2')
    plt.subplot(133),plt.imshow(img3),plt.title('3')
    plt.show()
    print 'done.'
