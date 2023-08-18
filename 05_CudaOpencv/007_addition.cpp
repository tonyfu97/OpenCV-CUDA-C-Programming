#include <iostream>
#include "opencv2/opencv.hpp"

int main()
{
    cv::Mat img1 = cv::imread("images/cameraman.tif", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("images/circles.png", cv::IMREAD_GRAYSCALE);

    cv::cuda::GpuMat d_img1, d_img2, d_result;
    cv::Mat h_result;
    d_img1.upload(img1);
    d_img2.upload(img2);

    cv::cuda::add(d_img1, d_img2, d_result);
    d_result.download(h_result);

    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    cv::imshow("result", h_result);

    cv::imwrite("images/007_addition.png", h_result);

    cv::waitKey();
    return 0;
}