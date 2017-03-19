
# install CUDA separately

# refresh and upgrade
sudo apt-get update
sudo apt-get upgrade

# developer tools
sudo apt-get install build-essential cmake pkg-config

# image libraries
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev

# video libraries
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev

# highgui module for advanced GUI
sudo apt-get install libgtk-3-dev

# Optimizations for math and matrices
sudo apt-get install libatlas-base-dev gfortran

# OTHER: ===================

# Parallelism and linear algebra libraries:
sudo apt-get install -y libtbb-dev libeigen3-dev

# Documentation:
sudo apt-get install -y doxygen

# Media I/O:
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev

# Video I/O:
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev

# ===================

# Python development headers
sudo apt-get install python2.7-dev python3.5-dev

# download OpenCV source
cd ~
# wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip
# unzip opencv.zip
wget https://github.com/opencv/opencv/archive/3.2.0.zip
unzip 3.2.0.zip

# Need the contrib repo for a full install
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip
unzip opencv_contrib.zip

# Left out-configuring the Python environment
cd ~/opencv-3.2.0/ 
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=ON \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
    -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
    -D BUILD_EXAMPLES=ON ..

# check to make sure we have CUDA on and video reading on before running make!

# compile
make -j4

# now, actually INSTALL
sudo make install
sudo ldconfig

# cleanup for python
cd /usr/local/lib/python3.5/site-packages/
sudo mv cv2.cpython-35m-x86_64-linux-gnu.so cv2.so
cd ~/.virtualenvs/cv/lib/python3.5/site-packages/
ln -s /usr/local/lib/python3.5/site-packages/cv2.so cv2.so

# Now, test by workon cv, python, import cv2