"""
        filter.py
        3/25/2017
        Nicholas S. Bradford

        http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html

"""

import numpy as np
from matplotlib import pyplot as plt


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [np.sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices


def particlefilter(sequence, pos, stepsize, n):
    seq = iter(sequence)
    x = np.ones((n, 2), int) * pos                   # Initial position
    # f0 = seq.next()[tuple(pos)] * ones(n)         # Target colour model
    f0 = next(seq)[tuple(pos)] * np.ones(n)         # Target colour model
    yield pos, x, np.ones(n)/n                       # Return expected position, particles and weights
    for im in seq:
        update = np.random.uniform(-stepsize, stepsize, x.shape).astype(int)  # Particle motion model: uniform step
        x += update
        x  = x.clip(np.zeros(2), np.array(im.shape)-1).astype(int) # Clip out-of-bounds particles
        f  = im[tuple(x.T)]                         # Measure particle colours
        w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
        w /= np.sum(w)                                 # Normalize w
        yield np.sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
        if 1./np.sum(w**2) < n/2.:                     # If particle cloud degenerate:
            x  = x[resample(w),:]                     # Resample particles according to weights


if __name__ == "__main__":
    # import pylab
    # from itertools import izip
    import time
    # pylab.ion()
    seq = [ im for im in np.zeros((20,240,320), int)]      # Create an image sequence of 20 frames long
    x0 = np.array([120, 160])                              # Add a square with starting position x0 moving along trajectory xs
    xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0
    for t, x in enumerate(xs):
        xslice = slice(x[0]-8, x[0]+8)
        yslice = slice(x[1]-8, x[1]+8)
        seq[t][xslice, yslice] = 255

    for im, p in zip(seq, particlefilter(seq, x0, 8.0, 100)): # Track the square through the sequence
        pos, xs, ws = p
        position_overlay = np.zeros_like(im)
        pos_tup = tuple(pos.astype(int))
        print(tuple(pos))
        
        position_overlay[pos_tup[0], pos_tup[1]] = 1
        # position_overlay[120, 160] = 1
        particle_overlay = np.zeros_like(im)
        particle_overlay[tuple(xs.T)] = 1
        # pylab.hold(True)
        # pylab.draw()
        time.sleep(0.3)
        # clf()                                           # Causes flickering, but without the spy plots aren't overwritten
        # pylab.imshow(im,cmap=pylab.cm.gray)                         # Plot the image
        # pylab.spy(position_overlay, marker='.', color='b')    # Plot the expected position
        # pylab.spy(particle_overlay, marker=',', color='r')    # Plot the particles
        # pylab.spy(np.eye(im.shape[0], im.shape[1]), marker='.', color='r')
        plt.imshow(position_overlay, interpolation='nearest')
        plt.imshow(particle_overlay, interpolation='nearest')
        plt.show()
    


# if __name__ == "__main__":
#     import pylab
#     # from itertools import izip
#     import time
#     pylab.ion()
#     seq = [ im for im in np.zeros((20,240,320), int)]      # Create an image sequence of 20 frames long
#     x0 = np.array([120, 160])                              # Add a square with starting position x0 moving along trajectory xs
#     xs = np.vstack((np.arange(20)*3, np.arange(20)*2)).T + x0
#     for t, x in enumerate(xs):
#         xslice = slice(x[0]-8, x[0]+8)
#         yslice = slice(x[1]-8, x[1]+8)
#         seq[t][xslice, yslice] = 255

#     for im, p in zip(seq, particlefilter(seq, x0, 8.0, 100)): # Track the square through the sequence
#         pos, xs, ws = p
#         position_overlay = np.zeros_like(im)
#         pos_tup = tuple(pos.astype(int))
#         print(tuple(pos))
        
#         position_overlay[pos_tup[0], pos_tup[1]] = 1
#         # position_overlay[120, 160] = 1
#         particle_overlay = np.zeros_like(im)
#         particle_overlay[tuple(xs.T)] = 1
#         pylab.hold(True)
#         pylab.draw()
#         time.sleep(0.3)
#         pylab.clf()                                           # Causes flickering, but without the spy plots aren't overwritten
#         pylab.imshow(im,cmap=pylab.cm.gray)                         # Plot the image
#         pylab.spy(position_overlay, marker='.', color='b')    # Plot the expected position
#         pylab.spy(particle_overlay, marker=',', color='r')    # Plot the particles
#         pylab.spy(np.eye(im.shape[0], im.shape[1]), marker='.', color='r')
#     show()

