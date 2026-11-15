#include "my_functions.hpp"
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    using Pixel = unsigned char; 

    inline int clamp_index(int value, int low, int high){
        return std::max(low, std::min(value, high));
    }

    void validate_gray_image(const cv::Mat& image, const char* function_name){
        if (image.empty()) {
            throw std::invalid_argument(std::string(function_name) + ": input image is empty");
        }
        if (image.type() != CV_8UC1) {
            throw std::invalid_argument(std::string(function_name) + ": input must be CV_8UC1 (8-bit grayscale)");
        }
    }


    // combines the vertical and horizontal gradients and then limits them to 255
    inline Pixel combine_sobel_gradients(int gx, int gy){
        const int abs_gx = std::min(std::abs(gx), 255);
        const int abs_gy = std::min(std::abs(gy), 255);
        return cv::saturate_cast<Pixel>(0.5 * (abs_gx + abs_gy));
    }

    //calculates the sobel result for one pixel
    //neighboring pixels in the middle row or column receive a weight of 2
    inline Pixel sobel_pixel(const Pixel* top, const Pixel* middle, const Pixel* bottom, int x, int left, int right){
        const int gx = (static_cast<int>(top[right]) + 2 * static_cast<int>(middle[right]) + static_cast<int>(bottom[right])) - (static_cast<int>(top[left]) + 2 * static_cast<int>(middle[left]) + static_cast<int>(bottom[left]));
        const int gy = (static_cast<int>(bottom[left]) + 2 * static_cast<int>(bottom[x]) + static_cast<int>(bottom[right])) - (static_cast<int>(top[left]) + 2 * static_cast<int>(top[x]) + static_cast<int>(top[right]));
        return combine_sobel_gradients(gx, gy);
    }


    inline void add_histogram_value(std::array<std::size_t, 256>& histogram, std::array<std::size_t, 16>& high_histogram, Pixel value){
        ++histogram[value];
        ++high_histogram[value >> 4];
    }

    inline void remove_histogram_value(std::array<std::size_t, 256>& histogram, std::array<std::size_t, 16>& high_histogram, Pixel value){
        --histogram[value];
        --high_histogram[value >> 4];
    }

    //finds the median using thew two histograms: fonds first the one group in the 16 where the pixel belongs, and then its exact place 
    inline Pixel histogram_median(const std::array<std::size_t, 256>& histogram, const std::array<std::size_t, 16>& high_histogram, std::size_t rank){
        std::size_t prefix = 0;
        int selected_high = 0;

        for (int high = 0; high < 16; ++high) {
            if (prefix + high_histogram[high] > rank) {
                selected_high = high;
                rank -= prefix;
                break;
            }
            prefix += high_histogram[high];
        }

        std::size_t low_prefix = 0;
        for (int low = 0; low < 16; ++low) {
            const std::size_t count = histogram[(selected_high << 4) + low];
            if (low_prefix + count > rank) {
                return static_cast<Pixel>((selected_high << 4) + low); //reconstructs the original pixel value
            }
            low_prefix += count;
        }

        throw std::logic_error("histogram_median: rank not found");
    }

}

//there are no more multiplications by 0, and the center unused pixel is gone
void my_sobel(const cv::Mat& im_in, cv::Mat& im_out){

    validate_gray_image(im_in, "my_sobel");

    //dimenstions of the picture
    const int rows = im_in.rows;
    const int cols = im_in.cols;
    im_out.create(rows, cols, CV_8UC1);

    //each row is being processed in a different cpu thread
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            //don't let the code access the rows outside of the image
            const int top_row = (r > 0) ? r - 1 : r;
            const int bottom_row = (r + 1 < rows) ? r + 1 : r;

            //direct pointers to image rows
            const Pixel* top = im_in.ptr<Pixel>(top_row);
            const Pixel* middle = im_in.ptr<Pixel>(r);
            const Pixel* bottom = im_in.ptr<Pixel>(bottom_row);
            Pixel* output = im_out.ptr<Pixel>(r);

            //left border: replicate column zero -> wasn't used on the original code
            output[0] = sobel_pixel(top, middle, bottom, 0, 0,(cols > 1) ? 1 : 0);

            if (cols > 2) {
                int x = 1;

                //four output pixels are calculated per iteration
                for (; x + 3 < cols - 1; x += 4) {
                    const int t0 = top[x - 1];
                    const int t1 = top[x];
                    const int t2 = top[x + 1];
                    const int t3 = top[x + 2];
                    const int t4 = top[x + 3];
                    const int t5 = top[x + 4];

                    const int m0 = middle[x - 1];
                    const int m1 = middle[x];
                    const int m2 = middle[x + 1];
                    const int m3 = middle[x + 2];
                    const int m4 = middle[x + 3];
                    const int m5 = middle[x + 4];

                    const int b0 = bottom[x - 1];
                    const int b1 = bottom[x];
                    const int b2 = bottom[x + 1];
                    const int b3 = bottom[x + 2];
                    const int b4 = bottom[x + 3];
                    const int b5 = bottom[x + 4];

                    // Sobel outputs reuse these values for their Gx gradients.
                    const int v0 = t0 + 2 * m0 + b0;
                    const int v1 = t1 + 2 * m1 + b1;
                    const int v2 = t2 + 2 * m2 + b2;
                    const int v3 = t3 + 2 * m3 + b3;
                    const int v4 = t4 + 2 * m4 + b4;
                    const int v5 = t5 + 2 * m5 + b5;

                    const int gx0 = v2 - v0;
                    const int gx1 = v3 - v1;
                    const int gx2 = v4 - v2;
                    const int gx3 = v5 - v3;

                    const int top_h0 = t0 + 2 * t1 + t2;
                    const int top_h1 = t1 + 2 * t2 + t3;
                    const int top_h2 = t2 + 2 * t3 + t4;
                    const int top_h3 = t3 + 2 * t4 + t5;

                    const int bottom_h0 = b0 + 2 * b1 + b2;
                    const int bottom_h1 = b1 + 2 * b2 + b3;
                    const int bottom_h2 = b2 + 2 * b3 + b4;
                    const int bottom_h3 = b3 + 2 * b4 + b5;

                    const int gy0 = bottom_h0 - top_h0;
                    const int gy1 = bottom_h1 - top_h1;
                    const int gy2 = bottom_h2 - top_h2;
                    const int gy3 = bottom_h3 - top_h3;

                    output[x]     = combine_sobel_gradients(gx0, gy0);
                    output[x + 1] = combine_sobel_gradients(gx1, gy1);
                    output[x + 2] = combine_sobel_gradients(gx2, gy2);
                    output[x + 3] = combine_sobel_gradients(gx3, gy3);
                }

                // loop for the interiror 3 pixels
                for (; x < cols - 1; ++x) {
                    output[x] = sobel_pixel(top, middle, bottom, x, x - 1, x + 1);
                }

                //right border: replicate the last column -> wasn't used on the original code 
                output[cols - 1] = sobel_pixel(top, middle, bottom,cols - 1, cols - 2, cols - 1);
            } else if (cols == 2) {
                // if it has only two columns, both pixels are border pixels 
                output[1] = sobel_pixel(top, middle, bottom, 1, 0, 1);
            }
        }
    });
}


/*
Median: slower than sobel
for kernels 3x3 and 5x5: functio nth_element and row pointers
for kernels 7x7 and above: less demanding strategy. Finds the median without having to load all the pixels all the time 
output rows are independent
*/
void my_median(const cv::Mat& im_in, cv::Mat& im_out, int n){
    validate_gray_image(im_in, "my_median");

    if (n < 3 || (n % 2) == 0 || n > 255) { 
        throw std::invalid_argument("my_median: kernel size must be odd and between 3 and 255");
    }

    const int rows = im_in.rows;
    const int cols = im_in.cols;
    const int half = n / 2;
    const std::size_t k = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t median_rank = k / 2;

    im_out.create(rows, cols, CV_8UC1);

    if (n <= 5) {
        // nth_element has average linear complexity in the number of samples.
        // Allocate one scratch vector per parallel worker and reuse it for all
        // output pixels in the worker's assigned row range.
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            std::vector<Pixel> window(k);

            for (int r = range.start; r < range.end; ++r) {
                Pixel* output = im_out.ptr<Pixel>(r);

                for (int c = 0; c < cols; ++c) {
                    std::size_t p = 0;

                    for (int dy = -half; dy <= half; ++dy) {
                        const int rr = clamp_index(r + dy, 0, rows - 1);
                        const Pixel* source_row = im_in.ptr<Pixel>(rr);

                        for (int dx = -half; dx <= half; ++dx) {
                            const int cc = clamp_index(c + dx, 0, cols - 1);
                            window[p++] = source_row[cc];
                        }
                    }

                    std::nth_element(window.begin(), window.begin() + median_rank, window.end());
                    output[c] = window[median_rank];
                }
            }
        });
        return;
    }

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int r = range.start; r < range.end; ++r) {
            Pixel* output = im_out.ptr<Pixel>(r);

            std::vector<const Pixel*> window_rows(static_cast<std::size_t>(n));
            for (int dy = -half; dy <= half; ++dy) {
                const int rr = clamp_index(r + dy, 0, rows - 1);
                window_rows[static_cast<std::size_t>(dy + half)] =
                    im_in.ptr<Pixel>(rr);
            }

            std::array<std::size_t, 256> histogram{};
            std::array<std::size_t, 16> high_histogram{};

            // Initialize the n*n window for output column zero, including replicated samples to the left of the image
            for (int dy = 0; dy < n; ++dy) {
                const Pixel* source_row = window_rows[static_cast<std::size_t>(dy)];
                for (int dx = -half; dx <= half; ++dx) {
                    const int cc = clamp_index(dx, 0, cols - 1);
                    add_histogram_value(histogram, high_histogram, source_row[cc]);
                }
            }

            for (int c = 0; c < cols; ++c) {
                output[c] = histogram_median(histogram, high_histogram,median_rank);

                if (c + 1 < cols) {
                    // Transition from window centered at c to c+1.
                    const int outgoing_col = clamp_index(c - half, 0, cols - 1);
                    const int incoming_col = clamp_index(c + half + 1, 0, cols - 1);

                    for (int dy = 0; dy < n; ++dy) {
                        const Pixel* source_row = window_rows[static_cast<std::size_t>(dy)];
                        remove_histogram_value(histogram, high_histogram, source_row[outgoing_col]);
                        add_histogram_value(histogram, high_histogram,source_row[incoming_col]);
                    }
                }
            }
        }
    });
}
