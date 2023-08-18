#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat h_img1 = cv::imread("images/cameraman.tif", cv::IMREAD_GRAYSCALE);
    cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_result5, d_img1;
    d_img1.upload(h_img1);

    cv::cuda::threshold(d_img1, d_result1, 128, 255, cv::THRESH_BINARY);
    cv::cuda::threshold(d_img1, d_result2, 128, 255, cv::THRESH_BINARY_INV);
    cv::cuda::threshold(d_img1, d_result3, 128, 255, cv::THRESH_TRUNC);
    cv::cuda::threshold(d_img1, d_result4, 128, 255, cv::THRESH_TOZERO);
    cv::cuda::threshold(d_img1, d_result5, 128, 255, cv::THRESH_TOZERO_INV);

    cv::Mat h_result1, h_result2, h_result3, h_result4, h_result5;
    d_result1.download(h_result1);
    d_result2.download(h_result2);
    d_result3.download(h_result3);
    d_result4.download(h_result4);
    d_result5.download(h_result5);

    cv::Mat combined(h_img1.rows, 5 * h_img1.cols, h_img1.type());
    h_result1.copyTo(combined(cv::Rect(0 * h_img1.cols, 0, h_img1.cols, h_img1.rows)));
    h_result2.copyTo(combined(cv::Rect(1 * h_img1.cols, 0, h_img1.cols, h_img1.rows)));
    h_result3.copyTo(combined(cv::Rect(2 * h_img1.cols, 0, h_img1.cols, h_img1.rows)));
    h_result4.copyTo(combined(cv::Rect(3 * h_img1.cols, 0, h_img1.cols, h_img1.rows)));
    h_result5.copyTo(combined(cv::Rect(4 * h_img1.cols, 0, h_img1.cols, h_img1.rows)));

    // Add titles
    cv::putText(combined, "Binary", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Binary Inv", cv::Point(h_img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Trunc", cv::Point(2 * h_img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "To Zero", cv::Point(3 * h_img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "To Zero Inv", cv::Point(4 * h_img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Combined Results", combined);
    cv::imwrite("images/012_thresholding_combined.png", combined);

    cv::waitKey();
    return 0;
}
