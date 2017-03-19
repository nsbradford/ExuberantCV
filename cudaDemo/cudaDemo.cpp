// #include <iostream>
// #include "opencv2/opencv.hpp"
// #include "opencv2/gpu/gpu.hpp"

// int main (int argc, char* argv[])
// {
//     try
//     {
//         cv::Mat src_host = cv::imread("file.png", CV_LOAD_IMAGE_GRAYSCALE);
//         cv::gpu::GpuMat dst, src;
//         src.upload(src_host);

//         cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

//         cv::Mat result_host;
//         dst.download(result_host);

//         cv::imshow("Result", result_host);
//         cv::waitKey();
//     }
//     catch(const cv::Exception& ex)
//     {
//         std::cout << "Error: " << ex.what() << std::endl;
//     }
//     return 0;
// }

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace cv;

int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("../img/grass.jpg", cv::IMREAD_GRAYSCALE);
        cv::imshow("Source", src_host);
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);
        cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat result_host(dst);
        cv::imshow("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}