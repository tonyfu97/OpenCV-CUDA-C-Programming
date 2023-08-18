#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img1 = cv::imread("images/autumn.tif", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_img1;

    d_img1.upload(img1);

    cv::cuda::cvtColor(d_img1, d_result1, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(d_img1, d_result2, cv::COLOR_BGR2HSV);
    cv::cuda::cvtColor(d_img1, d_result3, cv::COLOR_BGR2YCrCb);
    cv::cuda::cvtColor(d_img1, d_result4, cv::COLOR_BGR2XYZ);

    cv::Mat h_result1, h_result2, h_result3, h_result4;
    d_result1.download(h_result1);
    d_result2.download(h_result2);
    d_result3.download(h_result3);
    d_result4.download(h_result4);

    // Convert grayscale to color for correct concatenation
    cv::cvtColor(h_result1, h_result1, cv::COLOR_GRAY2BGR);

    // Concatenate the images
    cv::Mat combined(img1.rows, 5 * img1.cols, img1.type());
    img1.copyTo(combined(cv::Rect(0 * img1.cols, 0, img1.cols, img1.rows)));
    h_result1.copyTo(combined(cv::Rect(1 * img1.cols, 0, img1.cols, img1.rows)));
    h_result2.copyTo(combined(cv::Rect(2 * img1.cols, 0, img1.cols, img1.rows)));
    h_result3.copyTo(combined(cv::Rect(3 * img1.cols, 0, img1.cols, img1.rows)));
    h_result4.copyTo(combined(cv::Rect(4 * img1.cols, 0, img1.cols, img1.rows)));

    // Add titles to the combined image
    cv::putText(combined, "Original Image", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::putText(combined, "Grayscale", cv::Point(img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::putText(combined, "HSV", cv::Point(2 * img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::putText(combined, "YCrCb", cv::Point(3 * img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
    cv::putText(combined, "XYZ", cv::Point(4 * img1.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

    cv::imshow("Combined Results", combined);
    cv::imwrite("images/011_color_space_conversion_combined.png", combined);

    cv::waitKey();
    return 0;
}
