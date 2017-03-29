"""
    model.py
    27 March 2017
    Nicholas S. Bradford

"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from .config import Constants
from .plotter import plotModel

class State():

    def __init__(self, model1, model2=None, img=None):
        self.model1 = model1
        self.model2 = model2
        self.img = img


class LineModel():
    """ Represents a linear hypothesis for a single lane. 
        Attributes:
            offset (float): shortest (perpendicular) distance to the lane in meters
            orientation (float): that the lane is offset from dead-ahead (+ slope means + degrees)
    """

    OFFSET_MIN = -5.0
    OFFSET_MAX = 5.0
    OFFSET_RANGE = OFFSET_MAX - OFFSET_MIN
    ORIENTATION_MIN = 90.0
    ORIENTATION_MAX = 270.0
    ORIENTATION_RANGE = ORIENTATION_MAX - ORIENTATION_MIN
    CENTER = Constants.IMG_SCALED_WIDTH / 2.0
    NOSE_HEIGHT = Constants.IMG_CUTOFF

    def __init__(self, offset, orientation, m=None, b=None, height=Constants.IMG_SCALED_HEIGHT, 
                        width=Constants.IMG_SCALED_WIDTH, widthInMeters=Constants.IMG_WIDTH_IN_METERS):
        self.offset = offset
        self.orientation = orientation
        offset_pix = LineModel.pixelsToMeters(self.offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        if not m and not b:
            m, b = LineModel.offsetOrientationtoLine(offset, orientation)
        self.m = m
        self.b = b

    @classmethod
    def from_line(cls, m, b):
        offset, orientation = LineModel.lineToOffsetOrientation(m, b)
        return cls(offset, orientation + 180, m=m, b=b)

    @staticmethod
    def lineToOffsetOrientation(m, b):
        """ 
            Args:
                TODO
        """
        pixel_offset = LineModel.perpendicularDistancePixels(x0=LineModel.CENTER, 
                            y0=LineModel.NOSE_HEIGHT, slope=m, intercept=b)
        offset = LineModel.pixelsToMeters(pixel_offset, pixel_width=Constants.IMG_SCALED_WIDTH, 
                            meters_width=Constants.IMG_WIDTH_IN_METERS)
        raw_orientation = math.degrees(math.atan(m))
        angle_offset = - 90 if raw_orientation >= 0 else 90
        orientation = raw_orientation + angle_offset
        return offset, orientation

    @staticmethod
    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ First, convert [y=mx+b] to [ax+by+c=0]
            f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 
        """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    @staticmethod
    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        """
            Args:
                pixel_offset: offset from lane, in img pixels
                pixel_width: width of image in pixels
                meters_width: width of image in 
        """
        meters_per_pix = meters_width / pixel_width
        return pixel_offset * meters_per_pix

    @staticmethod
    def metersToPixels(meter_offset, pixel_width, meters_width):
        """
            Args:
                pixel_offset: offset from lane, in img pixels
                pixel_width: width of image in pixels
                meters_width: width of image in 
        """
        meters_per_pix = meters_width / pixel_width
        return meter_offset / meters_per_pix

    @staticmethod
    def offsetOrientationtoLine(offset, raw_orientation):
        angle_offset = 90#- 90 if raw_orientation >= 0 else 90
        orientation = raw_orientation + angle_offset
        m = math.tan(math.radians(orientation))
        b = LineModel.calcIntercept(x0=LineModel.CENTER, y0=LineModel.NOSE_HEIGHT, slope=m, 
                            perp_distance=offset)
        return m, b

    @staticmethod
    def calcIntercept(x0, y0, slope, perp_distance):
        first_term = perp_distance * (math.sqrt(slope**2 + (-1)**2))
        second_term = (- slope * x0 + y0)
        sign = LineModel.calcInterceptSign(perp_distance, slope)
        return sign * first_term + second_term

    @staticmethod
    def calcInterceptSign(perp_distance, slope):
        answer = 1
        positive_slope = slope >= 0
        positive_offset = perp_distance >= 0
        if positive_slope and positive_offset:
            answer = -1
        elif positive_slope and not positive_offset:
            answer = 1
        elif not positive_slope and positive_offset:
            answer = 1
        else: # not positive_slope and not positive_offset
            answer = -1
        return answer



class ParticleFilterModel():

    LEARNING_RATE = 1e-2
    VAR_OFFSET = 0.05
    VAR_ORIENTATION = 2.0
    VISUALIZATION_SIZE = 300
    

    def __init__(self, n=1000):
        self.state_size = 2
        self.n = n
        self.particles = self.init_particles()
        self.weights = self.init_weights()
        self.state_matrix = self.init_state()
        self.state = None
        self.last_measurement = None
        self.last_img = None
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        assert self.state_matrix.shape == (self.state_size,), self.state_matrix.shape


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

    @staticmethod
    def choose_between_models(state, last_measurement):
        m1 = state.model1
        m2 = state.model2
        if state.model2 is not None and last_measurement is not None:
            observations = np.array([   [m1.offset, m1.orientation],
                                        [m2.offset, m2.orientation]])
            distance = ParticleFilterModel.distance(new_particles=observations, measurement=last_measurement)
            print('\t\tChoice 1 dist {0:.2f}: \t*offset {1:.2f} \t orientation {2:.2f}'.format(distance[0], m1.offset, m1.orientation))
            print('\t\tChoice 2 dist {0:.2f}: \t*offset {1:.2f} \t orientation {2:.2f}'.format(distance[1], m2.offset, m2.orientation))
            if distance[0] > distance[1]:
                m1, m2 = m2, m1
        return m1, m2

    def update_state(self, state_measurement):
        if not state_measurement:
            return self.state_to_model() # TODO should still update somewhat
        self.last_img = state_measurement.img
        model1, model2 = ParticleFilterModel.choose_between_models(state_measurement, self.last_measurement)

        measurement = np.array([model1.offset, model1.orientation])
        self.last_measurement = measurement
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
        self.particles = self.apply_control(resampled_particles)
        self.weights = 1 /( 1 + ParticleFilterModel.distance(self.particles, measurement))
        self.weights /= np.sum(self.weights)
        self.state_matrix = self.calc_state()
        # print(np.std(self.particles[:, 0]), np.std(self.particles[:, 1]))
        assert self.particles.shape == (self.n,self.state_size), self.particles.shape
        assert self.weights.shape == (self.n,), self.weights.shape
        assert self.state_matrix.shape == (self.state_size,), self.state_matrix.shape
        self.state = self.matrix_to_state()
        return self.state


    @staticmethod
    def distance(new_particles, measurement):
        """ Squared distance. average of 2 columns, for each row """
        # TODO normalize by variance
        return ((new_particles - measurement) ** 2).mean(axis=1) * ParticleFilterModel.LEARNING_RATE

    def apply_control(self, resampled_particles):
        noise1 = np.random.normal(0, ParticleFilterModel.VAR_OFFSET, (self.n, 1))
        noise2 = np.random.normal(0, ParticleFilterModel.VAR_ORIENTATION, (self.n, 1))
        noise = np.hstack((noise1, noise2))
        return resampled_particles + noise

    def calc_state(self):
        return np.average(a=self.particles, axis=0, weights=self.weights) # for each column

    def matrix_to_state(self):
        model1 = LineModel(offset=self.state_matrix[0], orientation=self.state_matrix[1],
                            height=Constants.IMG_SCALED_HEIGHT, width=Constants.IMG_SCALED_WIDTH)
        # model2 = LineModel(offset=self.state_matrix[2], orientation=self.state_matrix[3],
        #                     height=Constants.IMG_SCALED_HEIGHT, width=Constants.IMG_SCALED_WIDTH)
        return State(model1, None)

    def show(self, img):
        print('\tFilter | \t offset {0:.2f} \t orientation {1:.2f}'.format(
                            self.state_matrix[0], self.state_matrix[1]))
        length = ParticleFilterModel.VISUALIZATION_SIZE
        img_shape = (length,length)
        shape = (LineModel.OFFSET_RANGE, LineModel.ORIENTATION_RANGE)
        
        particle_overlay = np.zeros(img_shape)
        x = self.particles + np.array([- LineModel.OFFSET_MIN, - LineModel.ORIENTATION_MIN])
        x = x.clip(np.array([0, 0]), np.array(shape)-1) # Clip out-of-bounds particles
        transform = np.array([length/LineModel.OFFSET_RANGE, length/LineModel.ORIENTATION_RANGE])
        x = (x * transform).astype(int)
        particle_overlay[tuple(x.T)] = 1
        
        if self.last_measurement is not None:
            ycoord = int((self.state_matrix[0] - LineModel.OFFSET_MIN) * transform[0])
            xcoord = int((self.state_matrix[1] - LineModel.ORIENTATION_MIN) * transform[1])
            cv2.circle(particle_overlay, (xcoord, ycoord), radius=15, color=255) #color=(0,0,255))
        cv2.imshow('particles', particle_overlay)

        if img is not None:
            cv2.imshow('model', plotModel(img, self.state.model1, color=(0,0,255)))
