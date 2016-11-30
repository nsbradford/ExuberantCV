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
    * Stereo: having two views of a scene, and get shape from "motion" between two images. Structure and depth are inherently ambiguous from single views.
    * How do humans see in 3D? Perspective effects (parallel lines), shading, changing textures, focus/defocus, motion
    * Problem formulation for basic stereo geometry: two different cameras with different optical centers, image planes, and scene points
      * The challenge is to find the matching images in each of the two images (image point correspondences) so that we can find the depth of that point (we also must know the orientation of the cameras, or "pose" or "calibration")
      * Cameras separated by distance 'B', and distance is positive in left image and negative in right
    * Epipolar geometry: the camera geometry constrains where the corresponding image from one view must occur in the second view (lies along a pair of epipolar lines). The Epipoles are where the epipolar lines converge, and exist at infinity for parallel image planes
    * Stereo Correspondence
      * For each point, create a window, then travel across the epipolar lines in the other image and choose the best match with sum of squared error or cross-correlation. Window size can be an issue (esp. when there are multiple matches)
      * Occlusion: will not always find a region in the other image
      * Ordering constraint: regions should appear in the same order in both images, except in some odd cases (which are difficult to handle)
      * Modern methods: compared to Ground truth of depth, hoping for both a good match and smoothness
        * Optimize correspondence assignments jointly instead of pixel-by-pixel, basically can use Dynamic Time Warp on lines at a time
        * Describe it all as an Energy function
      * Challenges: low contrast/textureless image regions, occlusions, violations of brightness constancy (e.g. reflections)
  * 3C Camera calibration
    * Two transformations: from [arbitrary] world coordinate System to camera 3D coordinate system (extrinsic calibration or camera pose), and from the 3D coordinates in camera frame to the 2D image plane via projection (intrinsic parameters)
    * Extrinsic Camera calibration: 6 degrees of freedom.
      * Translation: can be expressed with matrix multiplication using homogeneous coordinates
      * Rotation: matrix column vectors are simply basic vectors of A frame expressed in terms of B frame,
        * Can be expressed as 3 different rotations (the order of application matters, and there are different standards)
        * Or, use homogeneous coordinates to just do a single multiplication (remember, rotation is not commutative like translation)
      * Extrinsic parameter matrix: We can combine the homogeneous translation and rotation into a single transformation matrix (which is invertible)
    * Intrinsic camera parameters: 5 DoFs
      * Real intrinsic parameters: must scale pixels to real-world units, and can get even uglier when considering that pixels may not be squares, and axes might not be at right angles
      * In an idealized world, 'f' focal length is only parameter in intrinsic param matrix
    * Combine intrinsic and extrinsic matrices into one matrix 'M' or 'pi' which transforms world point 'p' into homogenous coordinate in image, described by Translation of optical center from world coordinate origin, rotation R of the camera system, and focal length and aspect ratio (f, a) of camera
    * Calibrating cameras
      * Resectioning: setup some points in the world that we know the 3D locations of, and take some images. Given 6+ points (for 6 degrees of freedom), can use matrix eigenvalues to compute M.
      * Use Direct linear calibration (transformation): Use the SVD trick (singular value decomposition: express matrix A as UDV^T)
      * Geometric error function: find smallest distance between observed points and points predicted given M and the image points. Use Maximum Likelihood Estimation
      * Modern method: multi-plane calibration, basically by taking pictures of checkerboards from multiple angles. Off-the-shelf solutions: OpenCV supports directly
        * If you have such a checkerboard mounted on a robot arm, can actually automate the entire calibration process!
  * 3D Multiple views: mapping images to images
    * 2D Transformations: Translation and rotation (Euclidian/rigid body), similarity (changes size but not proportions), affine (preserves planes/straight lines and relative areas), and projective (preserves lines). The "Homography" is the full 8 degrees of freedom, and requires 4 points to discern between images.
    * Homographies and mosaics: can compute panoramas by stitching together images taken from same camera optical center, computing the transform, and transforming image #2 to overlap with the first
      * Content-based manipulation/coding: can be used with multiple images to, say, remove a tennis player from the scene and then add back in at will
      * Image rectification: use Homography to apply plane-to-plane transformations (warping), recovering planes in original proportions
        * Forward warping: send each pixel from original image into corresponding location in new image - but what if it doesn't exactly align with a pixel?
        * Inverse warping: for each pixel (location) in new image, go in the reverse direction to find where it came from, and use bilinear interpolation to smooth 
    * Projective Geometry
      * A line in the image is a plane of rays through the origin, represented as a 3-vector (just like a homogeneous vector for a line)
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
