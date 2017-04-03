// #include <stdio.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;

// int main(int argc, char** argv )
// {
//     if ( argc != 2 )
//     {
//         printf("usage: DisplayImage.out <Image_Path>\n");
//         return -1;
//     }

//     Mat image;
//     image = imread( argv[1], 1 );

//     if ( !image.data )
//     {
//         printf("No image data \n");
//         return -1;
//     }
//     namedWindow("Display Image", WINDOW_AUTOSIZE );
//     imshow("Display Image", image);

//     waitKey(0);

//     return 0;
// }

#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
    VideoCapture cap("../vid/taxi_intersect.mp4"); // open the default camera "../vid/flying_turn.avi"
    if(!cap.isOpened()){  // check if we succeeded
        std::cout << "error" << std::endl;
        return -1;
    }
    std::cout << "hey" << std::endl;
    Mat edges;
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        // cap >> frame; // get a new frame from camera
        cap.read(frame);
        std::cout << "frame shape " << frame.rows << " " << frame.cols << std::endl;
        if(frame.empty()){
            std::cout << "bye" << std::endl;
            break;
        }
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);

        if (cvWaitKey(33) == 27) {
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    std::cout << "exit" << std::endl;
    return 0;
}