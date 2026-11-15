#ifndef MY_FUNCTIONS_HPP
#define MY_FUNCTIONS_HPP

#include <opencv2/core.hpp>

// Both functions expect an 8-bit, single-channel (grayscale) input image.
// The output is allocated/resized by the function.
// Border policy: replicate the nearest valid input pixel.
void my_sobel(const cv::Mat& im_in, cv::Mat& im_out);
void my_median(const cv::Mat& im_in, cv::Mat& im_out, int n);

#endif // MY_FUNCTIONS_HPP
