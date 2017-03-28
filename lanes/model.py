"""
    model.py
    27 March 2017
    Nicholas S. Bradford

"""

import numpy as np
import math
from config import Constants


class State():

    def __init__(self, model1, model2=None):
        self.model1 = model1
        self.model2 = model2



class LineModel():
    """ Represents a linear hypothesis for a single lane. 
        Attributes:
            offset (float): shortest (perpendicular) distance to the lane in meters
            orientation (float): that the lane is offset from dead-ahead (+ slope means + degrees)
    """

    OFFSET_MIN = -10.0
    OFFSET_MAX = 10.0
    ORIENTATION_MIN = -180.0
    ORIENTATION_MAX = 180.0


    def __init__(self, m, b, height, width, widthInMeters=3.0):
        self.m = m
        self.b = b
        center = width / 2.0
        nose_height = Constants.IMG_CUTOFF
        pixel_offset = LineModel.perpendicularDistancePixels(x0=center, y0=nose_height, slope=m, intercept=b)
        self.offset = LineModel.pixelsToMeters(pixel_offset, pixel_width=width, meters_width=widthInMeters)
        raw_orientation = math.degrees(math.atan(m))
        offset = - 90 if raw_orientation >= 0 else 90
        self.orientation = raw_orientation + offset


    # @classmethod
    # def from_filter(offset, orientation):
    #     m = math.tan(math.radians(orientation))
    #     b = 
    #     return LineModel(m, b, height=Constants.IMG_SCALED_HEIGHT, width=Constants.IMG_SCALED_WIDTH)

    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ First, convert [y=mx+b] to [ax+by+c=0]
            f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 
        """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        return pixel_offset * meters_width / pixel_width



class ParticleFilterModel():

    def __init__(self, n=200):
        self.state_size = 2
        self.n = n
        self.particles = self.init_particles()
        self.weights = self.init_weights()
        self.state = self.init_state()
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        assert self.state.shape == (self.state_size,)


    def init_particles(self):
        p_offset = np.random.uniform(low=LineModel.OFFSET_MIN, high=LineModel.OFFSET_MAX, size=self.n)
        p_orientation = np.random.uniform(low=LineModel.ORIENTATION_MIN, high=LineModel.ORIENTATION_MAX, size=self.n)
        return np.vstack((p_offset, p_orientation)).T



    def init_weights(self):
        """ Initialize to a uniform distribution"""
        w = np.ones((self.n,))
        w /= np.sum(w)
        return w

    def init_state(self):
        return self.calc_state()

    def update_state(self, state_measurement):
        model = state_measurement.model1
        measurement = np.array([model.offset, model.orientation])
        return self.update(measurement)

    def update(self, measurement):
        """ Algorithm for Particle Filter:
            def f (S, U, Z): # S is particles, U is control, Z is measurement
                S' = empty set
                for i = 0 ... n: # each new particle
                    Sample J ~ {w} with replacement # weights of current S
                    Estimate x' ~ p(x' | U, s_j)
                    W' = p(z|x') # new particle weight is likelihood given estimate
                    S' = S' u {< x', w'>} # add new particle to set
                for i = 0 ... n: # for each particle,  normalize weights
                    W_i /= n

        """
        assert measurement.shape == (self.state_size,)
        resampled_indices = np.random.choice(a=self.n, size=self.n, replace=True, p=self.weights)
        resampled_particles = self.particles[resampled_indices, :]
        new_particles = self.apply_control(resampled_particles)
        self.weights = 1 /( 1 + ParticleFilterModel.distance(new_particles, measurement))
        self.weights /= np.sum(self.weights) # Normalize w
        
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        self.state = self.calc_state()
        assert self.state.shape == (self.state_size,)
        return self.state_to_model()


    @staticmethod
    def distance(new_particles, measurement):
        """ Squared distance """
        return ((new_particles - measurement) ** 2).mean(axis=1)

    def apply_control(self, resampled_particles):
        noise = np.random.normal(0, 1, resampled_particles.shape)
        return resampled_particles + noise


    def calc_state(self):
        return np.average(a=self.particles, axis=0, weights=self.weights) # for each column


    def state_to_model(self):
        return State(LineModel(m=self.state[0], b=self.state[1], height=Constants.IMG_SCALED_HEIGHT, 
                            width=Constants.IMG_SCALED_WIDTH))

