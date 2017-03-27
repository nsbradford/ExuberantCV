"""
	model.py
	27 March 2017
	Nicholas S. Bradford

"""

import math
from config import Constants

class ParticleFilterModel():

	def __init__(self):
		self.state = None
		self.particles = None # with associated importance weights
		self.control = None

	def update(self, measurement):
		self.state = measurement


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

    def perpendicularDistancePixels(x0, y0, slope, intercept):
        """ f((x0,y0), ax+by+c=0) -> |ax0 + by0 + c| / (a^2 + b^2)^1/2 """
        a = slope
        b = -1
        c = intercept
        return abs(a * x0 + b * y0 + c) / math.sqrt(a ** 2 + b ** 2)

    def pixelsToMeters(pixel_offset, pixel_width, meters_width):
        return pixel_offset * meters_width / pixel_width
