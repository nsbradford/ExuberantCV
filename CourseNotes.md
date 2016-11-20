Intro to CV (Udacity: Georgia Tech CS4476/6476)

1.  Introduction
*  Image Processing for Computer Vision
  1. 2A Linear image processing
    * Filtering
      * Convolution and correlation
      * Filters as templates
      * Edge detection
        * Canny edge detector
        * Laplacian operator (two-dimensional gradients)
  * 2B Model fitting
    * Hough transform
      * Line detection, circle detection
      * Generalized to arbitrary shapes
  * 2C Frequency domain analysis
    * Fourier transform
      * Basis sets of vectors (linear independence and spanning property)
      * Fourier Series -> transforms -> discrete transforms
        * Any periodic function can be expressed in sine/cosine
    * Fourier Transofmr and Convolution
      * Convolution in spatial domain == multiplication in frequency domain, and vice versa
        * Trick: for very large matrices, instead of convolving, can use FFT on each matrix, multiply, and then inverse FT (IFT) to create convolved image
    * Aliasing
      * Aliasing occurs when the number of samples is too low to reconstruct a frequency properly (Nyquist frequency is 2x highest frequency in the signal)
      * Connection to images: when shrinking an image, instead of merely throwing away every other pixel (causes aliasing), use Guassian to retain more information.
      * Campbell-Robson Contrast Sensitivity: eye sensitivity varies for different frequencies and contrasts, and this can be exploited for compression techniques (DCT)
3. Camera Models and Views
  1. 3A Camera models
    * Cameras and Images
      * Image formation: lens, aperture, focal point
      * Depth of field: How large an area is "in focus"? Controlled by aperture size; small aperture condenses ray angles and creates a large DOF (deep isntead of shallow)
      * Field of View (FOV) depends on focal length: large focal length -> large FOV
      * Zooming and moving the camera create very different perspectives (e.g. portraits should be shot from ~6-10 ft with small FOV instead of up close with a large FOV). Can use "Dolly Zoom" effect.
      * Geometric Distortion (bowed out or in): has mathematical solutions, can be done automatically knowing a given lens and camera
      * Chromatic Aberration: different colors travel through the lenses differently (PhotoShop has approximate methods for solving this too)
    * Perspective Imaging
      * Center of Projection (COP) at the origin, normal (x, y) and z pointing positively towards the camera (negatively towards the plane)
      * Use projection equations to model projections of rays
      * Homogeneous coordinates: add [1] to location vector to make projection transformations linear operations. Multiply the projection matrix (3x4) by the homogenous point vector (4x1), and convert back to 2D to produce coordinates (u, v) of coordinates in image given (x, y, z) and focal length (f: distance from center of projection to image plane). Intuition: points at half the focal length should be half the size
      * Parallel lines: cross in the image, unless they are exactly aligned with the plane
        * Sets of parallel lines on the same plane (e.g. the ground) lead to colinear vanishing points, called the horizon for that plane
      * Orthographic projection (or "parallel projection"): special case where (x, y, z) -> (x, y)
      * Weak perspective: the change in depth of an object is insignificant compared from the distance to it from the camera.
  * 3B Stereo geometry
    * Stereo: having two views of a scene. Structure and depth are inherently ambiguous from single views.
    * How do humans see in 3D? 

  * 3C Camera calibration
  * 3D Multiple views
4. Image Features
  1. 4A Feature detection
  * 4B Feature descriptors
  * 4C Model fitting
5. Lighting
  1. 5A Photometry
  * 5B Lightness
  * 5C Shape from shading
6.  Image Motion
  1. 6A Overview
  * 6B Optical flow
7.  Tracking
  1. 7A Introduction to tracking
  * 7B Parametric models
  * 7C Non-parametric models
  * 7D Tracking considerations
8.  Classification and Recognition
  1. 8A Introduction to recognition
  * 8B Classification: Generative models
  * 8C Classification: Discriminative models
  * 8D Action recognition
9.  Useful Methods
  1. 9A Color spaces and segmentation
  * 9B Binary morphology
  * 9C 3D perception
10. Human Visual System
  1. 10A The retina
  * 10B Vision in the brain
