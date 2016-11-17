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
  * 3B Stereo geometry
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
