# Chapter 05: Getting Started with OpenCV and CUDA - Learning Reflections

**Author**: Tony Fu  
**Date**: August 17, 2023  
**Hardware and Software Configurations**: See [README.md](../README.md) at the repo root

**Reference**: Chapter 5 of [*Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA*](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA.git) by Bhaumik Vaidya.

## Core Concepts

## Issue with OpenCV

**Warning**: I encounted a build issue with OpenCV.

While attempting to build the project, I encountered the following error:
```
Build started...
1>------ Build started: Project: 05_CudaOpencv, Configuration: Release x64 ------
1>007_addition.cpp
1>C:\Users\Tony\opencv\include\opencv2\opencv.hpp(48,10): fatal error C1083: Cannot open include file: 'opencv2/opencv_modules.hpp': No such file or directory
1>Done building project "05_CudaOpencv.vcxproj" -- FAILED.
========== Build: 0 succeeded, 1 failed, 0 up-to-date, 0 skipped ==========
========== Build started at 10:30 PM and took 00.464 seconds ==========
```
The file `opencv_modules.hpp` is missing, leading me to suspect a corrupted build.
**Solution**: Currently under investigation.

### 1. Installing OpenCV with CUDA Support on Windows

The standard OpenCV prebuilt available on the official website does not include CUDA support. We will need to build OpenCV from source with CUDA enabled:

1. **Install CUDA Toolkit**: This step has already been done in Chapter 1.
2. **Install CMake**: CMake should already be installed. Use the `cmake --version` command to check.
3. **Clone the OpenCV and OpenCV_contrib Repositories**:
    ```bash
    cd C:\Users\YourUsername
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    ```
4. **Navigate to the OpenCV Directory and Create a Build Folder Inside It**:
    ```
    cd opencv
    mkdir build
    ```

5. **Run CMake to Configure the Build**:
    ```
    cmake -DOPENCV_EXTRA_MODULES_PATH=C:/Users/YourUsername/opencv_contrib/modules -DWITH_CUDA=ON -G "Visual Studio 17 2022" ..
    ```
Adjust the above command with the correct path to opencv_contrib and for your Visual Studio version.

6. **Open the Generated Visual Studio Solution File (OpenCV.sln) Inside the Build Directory**: This will open a Visual Studio window.

7. **Build the Solution**: In the toolbar near the top of the window, you'll see a dropdown box that allows you to select the configuration. It may be set to "Debug" by default. Click on this dropdown and select "Release." On Inside the solution explorer, open the "CMakeTargets" folder, right-click the "ALL_BUILD" project, and select 'Build'. This build took about two hour on my machine. Here is the build output:
    ```
    211>------ Build started: Project: ALL_BUILD, Configuration: Release x64 ------
   ========== Build: 211 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========
   ========== Build started at 6:48 PM and took 02:06:12.791 hours ==========
    ```
8. 

### 2. Basic OpenCV concepts

Here is a list of basic OpenCV concepts:

1. **Read an Image**:
   - **Color**: `cv::imread("image.jpg", cv::IMREAD_COLOR)`
   - **Black and White**: `cv::imread("image.jpg", cv::IMREAD_GRAYSCALE)`

2. **Define a Matrix**:
   - **8-bit single channel**: `cv::Mat mat(rows, cols, CV_8UC1)`
   - **8-bit three channel (e.g., BGR)**: `cv::Mat mat(rows, cols, CV_8UC3)`
   - **32-bit float**: `cv::Mat mat(rows, cols, CV_32F)`

3. **Display an Image**: `cv::imshow("Window Name", image)`

4. **Create and Destroy Windows**:
   - **Create a Named Window**: `cv::namedWindow("Window Name", flags)`
   - **Destroy a Window**: `cv::destroyWindow("Window Name")`

5. **Wait for a Key Press**: `cv::waitKey(delay)` - Waits for a pressed key, delay in milliseconds.

6. **Draw Shapes**:
   - **Line**: `cv::line(image, start_point, end_point, color, thickness)`
   - **Rectangle**: `cv::rectangle(image, top_left, bottom_right, color, thickness)`
   - **Circle**: `cv::circle(image, center, radius, color, thickness)`

7. **Add Text to an Image**: `cv::putText(image, text, org, fontFace, fontScale, color, thickness)`

8. **Save an Image**: `cv::imwrite("output.jpg", image)`

9. **Capture Video from a Camera**:
   - **Open a Camera**: `cv::VideoCapture cap(deviceID);`
   - **Read Frame**: `cap >> frame;` or `cap.read(frame);`

10. **Saving Video**:
   - **Create Video Writer**: `cv::VideoWriter writer("output.avi", codec, fps, frameSize, isColor)`
   - **Write Frame**: `writer << frame;`

### 3. Color Space Conversion

Here's a table that includes some of the common color spaces:

| Color Space  | Description                                                                                                        | OpenCV Macro            |
|--------------|--------------------------------------------------------------------------------------------------------------------|-------------------------|
| RGB          | Standard Red-Green-Blue color space. Most common in digital images.                                                 |                         |
| BGR          | Like RGB but channels are in reverse order. Default in OpenCV.                                                     | `cv::COLOR_BGR2*`       |
| Gray         | Grayscale representation.                                                                                          | `cv::COLOR_BGR2GRAY`    |
| HSV          | Hue, Saturation, Value. Represents colors in terms of hue (color), saturation (vividness), and value (brightness). | `cv::COLOR_BGR2HSV`     |
| HLS          | Hue, Lightness, Saturation. Similar to HSV but with a different representation of luminance.                       | `cv::COLOR_BGR2HLS`     |
| YCrCb        | Luma (Y) and chroma (Cr, Cb) representation. Separates image luminance from chrominance.                            | `cv::COLOR_BGR2YCrCb`   |
| XYZ          | CIE 1931 color space. Useful for colorimetry and color science.                                                     | `cv::COLOR_BGR2XYZ`     |
| Lab          | CIE Lab color space. Separates color into luminance (L) and color channels (a and b).                               | `cv::COLOR_BGR2Lab`     |
| Luv          | CIE Luv color space. Alternative to Lab with different color characterization.                                      | `cv::COLOR_BGR2Luv`     |
| YUV          | Luma (Y) and chrominance (U, V) representation. Used in video compression.                                          | `cv::COLOR_BGR2YUV`     |

Here's how to convert images to different color spaces with CUDA support:

1. Including Necessary Libraries
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
```
The necessary headers are included for using OpenCV and CUDA functionality.

2. Reading Image
```cpp
cv::Mat img1 = cv::imread("images/autumn.tif", cv::IMREAD_COLOR);
```
The image `autumn.tif` is read in color format into a CPU-based OpenCV `cv::Mat`.

3. CUDA Memory Allocation
```cpp
cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_img1;
```
Several GPU-based matrices are declared using OpenCV's `cv::cuda::GpuMat`. These will hold the CUDA processed data.

4. Uploading Image to GPU
```cpp
d_img1.upload(img1);
```
The original image is uploaded to the GPU memory for processing using the `upload` method.

5. Applying CUDA Color Conversions
```cpp
cv::cuda::cvtColor(d_img1, d_result1, cv::COLOR_BGR2GRAY);
cv::cuda::cvtColor(d_img1, d_result2, cv::COLOR_BGR2HSV);
cv::cuda::cvtColor(d_img1, d_result3, cv::COLOR_BGR2YCrCb);
cv::cuda::cvtColor(d_img1, d_result4, cv::COLOR_BGR2XYZ);
```
The image is converted to various color spaces (Grayscale, HSV, YCrCb, XYZ) using the CUDA-accelerated `cvtColor` method.

6. Downloading Results from GPU to CPU
```cpp
cv::Mat h_result1, h_result2, h_result3, h_result4;
d_result1.download(h_result1);
d_result2.download(h_result2);
d_result3.download(h_result3);
d_result4.download(h_result4);
```
The results are downloaded from the GPU memory to CPU memory using the `download` method.


### 4. Thresholding

| Method                 | Description                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| `cv::THRESH_BINARY`    | All pixels with a value greater than the threshold value are set to the max value (255 in this case), others to 0.  |
| `cv::THRESH_BINARY_INV`| The inverse of `cv::THRESH_BINARY`. Pixels greater than the threshold value are set to 0, others to the max value.   |
| `cv::THRESH_TRUNC`     | Pixel values greater than the threshold are truncated to the threshold value, and the lower values remain unchanged. |
| `cv::THRESH_TOZERO`    | Pixel values that are less than the threshold are set to 0, and those greater than or equal to it remain unchanged. |
| `cv::THRESH_TOZERO_INV`| The inverse of `cv::THRESH_TOZERO`. Pixels greater than the threshold are set to 0, others remain unchanged.         |
