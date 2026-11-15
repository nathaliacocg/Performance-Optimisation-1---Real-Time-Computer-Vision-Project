#include "my_functions.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline int clamp_index(int value, int lower, int upper)
{
    return std::max(lower, std::min(value, upper));
}

void validate_grayscale_input(const cv::Mat& image, const char* function_name)
{
    if (image.empty()) {
        throw std::invalid_argument(std::string(function_name) +
                                    ": input image is empty");
    }
    if (image.type() != CV_8UC1) {
        throw std::invalid_argument(std::string(function_name) +
                                    ": input must be CV_8UC1 (8-bit grayscale)");
    }
}

} // namespace

// Naive Sobel baseline, corrected for correctness but deliberately retaining
// the original column-major loop order (columns outside, rows inside) and
// Mat::at<> accesses. The row-major traversal and pointer-based access are
// optimization experiments for a later project stage.
// Correctness fixes:
// - replicate borders;
// - use the same norm as the OpenCV reference: 0.5*|Gx| + 0.5*|Gy|.
// The original multiplications by zero are intentionally retained here so
// this remains the naive baseline. Removing them is a later optimization.
void my_sobel(const cv::Mat& im_in, cv::Mat& im_out)
{
    validate_grayscale_input(im_in, "my_sobel");

    const int rows = im_in.rows;
    const int cols = im_in.cols;
    im_out.create(rows, cols, CV_8UC1);

    // Preserve the original naive traversal order for the baseline.
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            const int r0 = clamp_index(r - 1, 0, rows - 1);
            const int r2 = clamp_index(r + 1, 0, rows - 1);
            const int c0 = clamp_index(c - 1, 0, cols - 1);
            const int c2 = clamp_index(c + 1, 0, cols - 1);

            const int nw = im_in.at<unsigned char>(r0, c0);
            const int n  = im_in.at<unsigned char>(r0, c);
            const int ne = im_in.at<unsigned char>(r0, c2);
            const int w  = im_in.at<unsigned char>(r, c0);
            const int e  = im_in.at<unsigned char>(r, c2);
            const int sw = im_in.at<unsigned char>(r2, c0);
            const int s  = im_in.at<unsigned char>(r2, c);
            const int se = im_in.at<unsigned char>(r2, c2);

            const int center = im_in.at<unsigned char>(r, c);
            const int gx = 2 * e + ne + se - 2 * w - sw - nw
                         + 0 * n + 0 * center + 0 * s;
            const int gy = 2 * n + ne + nw - 2 * s - sw - se
                         + 0 * w + 0 * center + 0 * e;

            // Match the OpenCV reference exactly:
            // convertScaleAbs() first takes abs() and saturates each
            // gradient independently to CV_8U, then addWeighted() combines
            // the two 8-bit images with weights 0.5 and 0.5.
            const unsigned char abs_gx_8u =
                cv::saturate_cast<unsigned char>(std::abs(gx));
            const unsigned char abs_gy_8u =
                cv::saturate_cast<unsigned char>(std::abs(gy));

            const double magnitude =
                0.5 * abs_gx_8u + 0.5 * abs_gy_8u;

            im_out.at<unsigned char>(r, c) =
                cv::saturate_cast<unsigned char>(magnitude);
        }
    }
}

// Naive median baseline, corrected for the project comparison:
// - n must be odd and >= 3;
// - every pixel receives exactly n*n samples;
// - borders use replication;
// - the median is the zero-based element k/2.
void my_median(const cv::Mat& im_in, cv::Mat& im_out, int n)
{
    validate_grayscale_input(im_in, "my_median");

    if (n < 3 || (n % 2) == 0) {
        throw std::invalid_argument("my_median: kernel size n must be odd and >= 3");
    }

    const int rows = im_in.rows;
    const int cols = im_in.cols;
    const int half = n / 2;
    const std::size_t k = static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(n);

    im_out.create(rows, cols, CV_8UC1);
    std::vector<unsigned char> window(k);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::size_t p = 0;

            for (int dy = -half; dy <= half; ++dy) {
                const int rr = clamp_index(r + dy, 0, rows - 1);

                for (int dx = -half; dx <= half; ++dx) {
                    const int cc = clamp_index(c + dx, 0, cols - 1);
                    window[p++] = im_in.at<unsigned char>(rr, cc);
                }
            }

            std::sort(window.begin(), window.end());
            im_out.at<unsigned char>(r, c) = window[k / 2];
        }
    }
}
