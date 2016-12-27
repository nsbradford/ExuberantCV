Intro to CV (Udacity: Georgia Tech CS4476/6476)

1.  Introduction
*  Image Processing for Computer Vision
  1. 2A Linear image processing
    * Filtering
      * Convolution and correlation
      * Filters as templates
      * Edge detection: 
        * Canny edge detector: compute smoothed derivative of image, threshold to find regions of "significant" gradient, and then "thin" to get localized edge pixels
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
    * Fourier Transform and Convolution
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
      * Given any formula for points/lines, can switch the meanings between points and lines. So, given two points, simply take the cross product of their homogeneous coordinates to find the line between them. Can also find the intersection of two lines the same way.
      * Can extend all this logic easily into 3D projective geometry
    * Essential Matrix (precursor to fundamental matrix): can be useful for determining both the relative position and orientation between the cameras and the 3D position of corresponding image points.
      * Location of X in prime frame (X'), can be obtained by rotating and translating it (X' = RX + T)
      * X'.t E X = 0 (E is the essential matrix)
      * Parallel cameras:
        * R = Identity Matrix, and T = [=B, 0, 0].T where B is distance between the cameras. Thus, Essential Matrix E is written as T cross product R (which can be written as a multiplication). Make use of this to derive that epipolar lines will be horizontal
    * Fundamental Matrix: infer essential matrix properties from weakly calibrated cameras 
      * Write as P_image = K_internal * PHI_external * P_world
      * By estimating fundamental matrix from pixel correspondences in stereo images, can reconstruct epipolar geometry without intrinsic or extrinsic parameters: Line = F * point
      * Applications: stereo image rectification, photo synch (reconstruct using many images)
4. Image Features
  1. 4A Feature detection
    * Use local features (must detect and match) to find corresponding points between images and construct essential/fundamental matrices
    * Good features: repeatable/precise, saliency/matchability, compactness and efficiency, locality (a feature only covers a small portion of the image) 
    * Finding Corners: useful because gradients occur in both directions
      * Harris Corners: based on approximation and error model (intensity difference) over some small shift [u, v] and some window function (Gaussian, area around a point). Use a second-order (quadratic) Taylor expansion to compute for very small changes, and can cancel out second derivatives! Simplify to calculate a M="second moment matrix", such that E(u,v) ~= [u v] M [u v].T. Taking the eigenvalues of M, get the amount of error change in each direction. If both eigenvalues are large and similarly proportioned, then there are large gradients in both directions, and you've found a corner.
        * Step-by-step algorithm: compute gaussian derivatives at each pixel, compute second moment matrix M in Gaussian window around each pixel, compute corner response function R, threshold R, and find local maxima of response function
      * Other corners: Shi-Tomasi used by OpenCV cvGoodFeaturesToTrack(), Brown, and others
    * Scale Invariance: Harris corners are rotation invariant, mostly invariant to addititive/multiplication, but not to scale (if window is too small, may detect many curved edges instead of a corner).
      * Use kernel to find robust extrema (max/min) both in space and in scale (look at SIFT later). Key point localization involves taking the differences between Gaussian smoothings at different scales. Also look at "Harris-Laplacian" detector.
  * 4B Feature descriptors and SIFT: describe a point in order to match them between two images. Descriptor should be invariant (mostly unchanged between the images)
    * How about correlation? Not rotation invariant, also sensitive to photometric (brightness) changes, and normalized correlation is vulnerable to slight geometric changes and non-linear photometric changes
    * SIFT (Scale-Invariant Feature Transform): solves the Harris operator not being scale-invariant, and correlation not being rotation-invariant. Has both a detector (operator) and descriptor (important). Tricky to implement (a lot of details to get right)
      * 1) find keypoints (using Harris-Laplace or some other method), 2) assign orientation to each keypoint region by dominant gradient direction, 3) create a histogram of the gradients as a descriptor, 4) do that for 16 grid squares around the point -> creates a vector with 128 components that becomes the descriptor for the point
    * Matching feature points: use an approximation of K-Nearest-Neighbor called best-bin-first (modification of k-d tree algorithm, using a heap to identify bins by distance from a point, 100-1000x speedup over K-NN) with 95% accuracy (not practical to have a perfect algorithm)
      * Locality-sensitive hashing: try to have points in the original space map to same bin
      * Object recognition: train by extracting keypoints from reference image, then brute-force create affine transforms (requires 3 points) that matches several of the matching keypoints and compare against actual image for testing. This can work with part of the object (some keypoints) occluded
  * 4C Model fitting
    * Robust error functions
      * Feature-based alignment algorithm: after computing putative matches for descriptors for feature points, loop until happy: 1) Hypothesize transformation T from some matches, then 2) Verify transformation (search for other matches consistent with T)
        * Improvement to brute-force putative feature matching: set a threshold, and only pick matches that are above a certain threshold. Compute threshold with both 1-NN and 2-NN (nearly eliminates chance of best match being incorrect)
        * No matter what you do, there's a "distinctive vs invariant" competition
    * RANSAC (Random Sample Consensus): finding consistent matches between feature points, by randomly picking some points to define model, and then repeating until you find a model that matches many inliers (lie within some error bound threshold). Parameters: minimal set (dependent on model, which is a homography = 4 points), distance threshold for inliers, number of samples (such that there's a high probability p=0.99 that at least one random sample is free of outliers) set on the outlier ratio.
      * Cool thing - N will stay low for even a large number of outliers (e.g. > 50%) even when s is 8 (for fundamental matrix).
      * RANSAC picks the "closest" transform, allowing you to pick the points which come from that model (assuming some Gaussian noise) - then take the average of all those to compute the final transformation.
      * Algorithm for homography: Select 4 feature pairs at random, compute exact homography H, compute inliers where SSD (p_i', H * pi) < error threshold, keep largest set of inliers, and re-compute least-squares H estimate on all of the inliers
      * Note: error rate e is often unknown, so start at some high error rate and check to see if it works
      * Example: can use RANSAC to detect a plane in the image! (useful if on a flat runway...)
      * In conclusion: widely applicable, robust, easier to get working than Hough transform, really not good for approximate models (if the plane isn't really a plane), is used in nearly every robotic vision problem
5. Lighting
  1. 5A Photometry
    * Photometry: physics/graphics background
    * Radiometry (study of light): Radiance (L) is energy carried by a ray off of a surface in Watts per square meter per steradian, Irradiance is energy arriving at a surface from a direction in Watts per square meter
    * BRDF: bidrectional Reflectance Distribution Function relates Radiance and Irradiance
    * Reflection models: Diffuse/matte/body reflections (not shiny at all, scatters light), Specular/Surface/Glossy/Highlights reflections (very shiny, reflects light directly). Image intensity can be approximated as a combination of Body and Surface reflection
    * Body reflection: use Lambertian RBDF, which states that object will look the same from every direction
    * Surface reflection: source and sensor must have both tilt and rotation angles the exact same in order for the ray to hit the sensor
    * Combined model: "Phong" model
  * 5B Lightness
    * In general, images are very ambiguous, and we have to make many assumptions in order to pick from one of many possible solutions (e.g. lightness implies depth)
    * Assumption: light intensity varies slowly, reflection function is constant within an object, reflection function varies suddenly across different objects
    * How do we recover the "true" image given our lighting assumptions? Take derivative of log of image, threshold (high-pass filter), and then integrate to recover (losing the constant - absolute brightness unknown)
    * Color constancy: determine hue and saturation under different colors of lighting
    * Lightness constancy: gray-level reflectance under differing intensity of lighting
    * The white-balancing problem has some ok solutions for correcting light source to produce "real" image (works well for pictures), but still an open problem for CV/computational photography
  * 5C Shape from shading/lighting
    * Compute a Reflectance Map, assume everything is lambertian
    * Constained shape from shading approach: assume bounds are known, and object is integrable/smooth throughout with constant albedo (works poorly)
    * Photometric stereo: take several images with different lighting
6.  Image Motion
  1. 6A Overview: retrieve motion from discrete images: detect moving objects, detect boundaries between shots of video, segmentation of moving objects. 
    * Gestalt psychology covers motion in the human brain; an interesting finding is that we group together points that are moving similarly, and can use this from a [very] impoverished group of points for inference
    * Generally, have feature-based areas (good for large-motions) which provides sparse motion fields but robust tracking, and Direct/Dense methods ( good for ...) which recovers per-pixel motion from spatio-temporal brightness variationk
  * 6B Optical flow
    * Dense Flow: brightness constraint
      * Optic flow: apparent motion of objects or surfaces
      * Correspondence (Optic flow) problem: given pixel in Img(x, y, t), look for nearby pixels of same color in I(x, y, t+1). Combine these two assumptions to create the Brightness Constancy Constraint Equation.
        * Assumption: color constancy (brightness constancy for greyscale images), mathematically the difference between the two points is 0
        * Assumption: small motion (points don't move very far), can approximate with Taylor series expansion of gradients (assume gradients are same for 't' and 't+1')
      * Aperture problem: you can only tell motion locally when it's perpendicular to the edge
      * Smooth Optical flow: Adjust weighting factor balancing two error functions (based on the believed amount of noise in the video):
        * Minimize error generated by violating the Brightness Constancy Constraint Equation
        * Minimize error generated from changes in U and V (change in x and y) over the image, encouraging smoothness
    * Dense Flow: Lucas and Kanade
      * Start by assuming all u and v values are the same over some small 5x5 window... eventually produce equivalent of second moment matrix from Harris Corner Detector, where eigenvectors and eigenvalues of M related to edge direction and magnitude * With RGB, provides more accurate gradients (changes in intensity will be quite correlated between channels)
      * Iterative algorithm: 1) estimate each pixel's velocity with Lucas-Kanade equations, 2) warp I_t towards I_t+1 using estimated flow field, 3) repeat until convergence
        * Note that warping is often tricky to implement in practice
        * Often useful to low-pass filter images before estimating motion to allow for better derivative estimations
    * Hierarchical LK
      * Most common issue with finding flow is that pixels often move by a large amount (more than 1px)
      * Reduce the resolution! Create a Gaussian Pyramid of successively coarser images, then move downwards to get to finer images while running iterative LK at each level ("Laplacian Pyramid")
      * This works much better than non-hierarchical, but still fails for pixels at the occlusion boundaries
      * Spare LK (used by OpenCV): flow field only at corners (as opposed to dense flow field)
      * Modern CV builds a lot of extra layers on top of this
    * Motion Models
      * Can express general motion in a "simple" equation relating T(translation vector) and Omega(rotation) with focal length of camera (and other image properties)
      * Note that depth is only affected by translation, not rotation! This explains parallax as well (objects at different distances appear to move differently when translating)
      * If objects can be approximated as a plane, can simplify to an affine transform!
      * Motion Segmentation: By taking velocities of points across the immage and comparing them, it's possible to find foreground and background (the clustering is not trivial, however)
7.  Tracking
  1. 7A Introduction to tracking
    * Challenges: It's difficult to comptue optical flow everywhere, there can often be large displacements (rapid movement) may need to consider dynamics, and errors can compound (drift), occlusions (temporarily disappears) or disocclusions (object suddenly appears)
    * Shi Tomasi: incorporated into OpenCV
    * Tracking with dynamics key idea: given a model of expected motion, predict where objects will occur in next frame, then update model based on measurements. This allows us to restrict search, and improve measurement accuracy
    * Assumptions: motion is constant (objects don't disappear and reappear in totally different places), camera motion is continuous; thus, all changes should be gradual
  * 7B Parametric models
    * Tracking as Inference: model as a hidden state X (true params) and measurement Y (noisy observation of underlying state), and we get new information at every time step
      * Our Prior/Prediction/Belief (distribution) is combined with a noisy measurement (forms a distribution of "How likely is this measurement for each possible location?") to produce the Final Estimate, or Posterior Distribution
      * Can simplify with the assumption that only the previous state is necessary for predictions (include velocity, etc. in the state model)
      * Prediction equation: use Law of Total Probability/Marginalization: if you have a joint probability distribution, integrating over one variable gives you the distribution of the other
        * Use the independence assumption to then split into Dynamics (left) and Belief (right) models
        * P(X_t | y_0, ...) = Integral: P(X_t | X_t-1) P(X_t-1 | y_0, ...) dX_t-1 
      * Correction equation: combine Prediction estimate with observation model (get probability of getting the measurement you just received) using Bayes' Rule, in order to finally get the distribution
    * Kalman Filter
      * Linear dynamics model: state undergoes linear transformation plus Gaussian noise
      * Linear measurement model: measurement is linearly transformed state plus Gaussian noise
      * Steps: deterministic linear drift (according to model), stochastic diffusion (add noise), reactive effect of measurement (refine)
        * The new mean is a weighted average of prediction and measurement (higher weights for less uncertainty). Kalmain Gain is the weight given to the measurement
        * Note that there can still be issues when the predictions and measurement disagree slightly, because the combined result will have a very tight variance (despite the massive uncertainty). This is often indicative of an underlying issue with the model itself.
      * Cons: unimodal distribution means only one hypothesis, linear model restricts class of motion (can be fixed with "Extended Kalman Filter")
  * 7C Non-parametric models
    * Particule Filter: a way to describe non-Gaussian distributions by having the underlying model have arbitrary distribution. Density is represented by both location and weight of a particule. 
      * Assume we have some input, or "perturbation"
      * Given: prior probability of the sytem state, action (dynamical system) model, sensor model, and stream of observations
      * Bayes filter: general approach for estimating a distribution over time from incoming measurements and some process model
    * Particle Filters for Localization: robotics application
      * Use stochastic universal sampling/systematic resampling so that sampling is efficient
      * Other considerations: be efficient about resampling, make sure to overestimate noise, recover from failure by randomly distributing some extra particles at every step
    * Particle filters for real
      * Good paper: Condensation - Conditional Density Propagation for Visual Tracking (1997)
    * Tracking considerations: 
      * Mean-shift: find the modes of a distribution, given a set of samples by repeatedly moving in the direction of the center of mass of a local region (usually converges). Use a similarity function with the Bhattacharyya coefficient. Use a differentiable, isotropic, monotonically decreasing kernel (such as... the Gaussian!)
      * In order to track people (with segmented body parts), build a generic model, and then learn the appearance for that particular person using some examples
  * 7D Tracking considerations
    * Initialization: is tricky; can manually start, subtract background, or separate detector function
    * Sensor and Dynamics models: learn from data, or use domain knowledge; some form of ground truth often required
    * Prediction vs Correction trade-off: if dynamics model is too strong, incoming data will be ignored
    * Data Association: decide which measurements go with which objects. Simple strategy; only pay attention to the measurement closest to the prediction. Can be more sophisticated with multiple hypothesis
    * Drift is still an issue
8.  Classification and Recognition
  1. 8A Introduction to recognition
    * Object categorization: given a number of training images of a category, assign correct category label to new images
    * Issue: most objects fit into multiple categories (e.g. a German Shepherd is a dog is an animal). Use intuition from "Basic Level Categories" in humans. Dogs invoke a clear mental image and set of behaviors, while animal doesn't have a single type of image associated with it
      * Humans are suspected to have something like 10,000-30,000 object categories, on par with the number of nouns humans use in language
    * Challenges: training set has varying illumination, object pose, and clutter. Same image can mean different objects if seen in different contexts. Problem is very complex and computationally intensive (estimated to take up something on order of half of the brain in primates)
  * 8B Classification: Generative models
    * Supervised classification: given a collection of labeled examples, come up with a function that will predict the labels of new examples. 
      * Generative models: separately build models for each class, then for a new test input, compare the results of each model and return which is most confident; separately model conditional densities and priors
        * Firm probabilistic grounding, allows you to use priors, can use a small number of examples, new classes don't perturb previous models
        * Because of the probabilistic model, can be used to generate new examples
        * However, works best in lower-dimensional spaces, there are issues with acquiring the prior probability, the "hard" cases are difficult to model, and having lots of data won't improve the model.
      * Discriminative models: construct a decision boundary between the classes; model the posterior
      * "Risk" of a classifier strategy S is the expected loss; different mistakes may have different associated costs
      * Principal Component Analysis (PCA): all about which directions in the feature space have the greatest variance. Picture a 2D plot, and collapsing all points to the best-fitting line
        * Largest eigenvalue of calculated matrix will be in axis of least inertia (greatest variance)
        * Note that PCA only works with a single class/distribution, can can produce some weird results if run on multiple classes simultaneously.
        * Idea: complex objects like "faces" inhabit only a small subspace of all possible images. Thus, we want ot consturct a low-diensional linear subspace that best explains the variation in the set of face images. Side note: in general, there will be N-1 eigenvectors, because we're subtracting out the mean.
          * Can use a dimensionality trick to reduce the complexity of the calculations
          * Each huge-dimensional eigenvector produced can be seen as its own image of deviation from the "mean" image, and each will potentially represent some recognizable feature (e.g. lighting on one side of the face, etc.)
        * Project novel image into subspace using dot products and compare to closest training face (or optionally check reconstruction error); can really be seen this way as a generative model
        * Unfortunately, not robust to misalignment or background variation, and the direction of maximum variance doesn't always turn out to be good for classification between classes.
        * To combat possibility of nonlinear relationships in data, there are Nonlinear Kernel PCA methods.
      * Appearance-based tracking: build low-dimensional model before tracking starts, use model to find and track object
        * Problem: object appearance and environment are changing all the time, so have to handle pose variation, partial occlusion, illumination change, drifts, etc.
        * Combine particle filter, subspace-based detection, and incremental subspace updates to track changes in appearance. Trick is to make the update small enough that you don't start tracking some random object
        * Occlusion handling: compute a weight mask based on which pixels the model reconstructs well
  * 8C Classification: Discriminative models: create the best decision boundary between classes
    * With big data and massive dimensionality, boundaries become more interesting. A large part of work will be feature selection for how to discriminate between classes. Common assumption is that cost of mistakes (wrong assigned label) will be the same. Random forests are very popular nowadays
      * Windowing methods scan through entire image with different-sized windows computing a binary classifier
        * Ex) Summarize local distribution of gradients with histogram (like with SIFT)
        * Train a binary classifier. K-NN, Neural Networks, SVMs, Boosting, Random Forests...
    * Boosting and Face Detection
      * Iterate: construct weak learners, find which one performs best, and raise weights of training examples still being misclassified by weak learner. Final result is a linear combination of weak learners.
      * Face-detecting feature: use "integral image" with rectangle sliding windows approximating gradients
        * First few weak learners with AdaBoost (AYda-boost, not Ahda-boost) correspond to most significant features
      * Key idea: cascade filters, such that the first classifier does basic checks for if it's a face or not, and then the second, and finally the third searches for fine details - makes much more computationally efficient. End result is the Viola-Jones method, which is used in production extensively.
    * Support Vector Machines (SVMs): intuitively, find the optimal separating line between the classes (maximum margin).
      * Kernel trick for lifting things into a higher-dimensional space. (has some satisfiability constaints; function must be a dot product in high-dimensional space)
      * Gaussian RBF basically produces dot product in space with infinite dimenisons
      * Compute-heavy during training but very fast on test data
      * Cons: no "direct" multi-class SVMs (must do one-vs-all or one-vs-one combinations), selecting best kernel function is tricky in practice, 

  * 8D Action recognition
9.  Useful Methods
  1. 9A Color spaces and segmentation
  * 9B Binary morphology
  * 9C 3D perception
10. Human Visual System
  1. 10A The retina
  * 10B Vision in the brain
