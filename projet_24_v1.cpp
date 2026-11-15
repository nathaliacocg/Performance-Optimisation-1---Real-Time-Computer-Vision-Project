#include "my_functions.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

namespace {

struct Options {
    int kernel = 3;
    int width = 640;
    int height = 480;
    int warmup = 10;
    int frames = 100;
    int opencv_threads = 1;
    int camera_index = 0;
    bool use_video = false;
    bool display = true;
    std::string video_path;
    std::string csv_path = "timings.csv";
};

struct FrameTiming {
    double my_median_ms = 0.0;
    double my_sobel_ms = 0.0;
    double pipeline_ms = 0.0;
    double ref_median_ms = 0.0;
    double ref_sobel_ms = 0.0;
};

struct Difference {
    int differing_pixels = 0;
    double max_absolute_difference = 0.0;
};

long long monotonic_ns()
{
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
    }
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

double elapsed_ms(long long start_ns, long long end_ns)
{
    return static_cast<double>(end_ns - start_ns) / 1.0e6;
}

void print_usage(const char* program)
{
    std::cerr
        << "Usage: " << program << " [kernel] [options]\n"
        << "Options:\n"
        << "  --kernel N       Median kernel (odd, >= 3; default 3)\n"
        << "  --video FILE     Read a recorded video instead of camera 0\n"
        << "  --camera N       Camera index (default 0)\n"
        << "  --width W        Requested camera width (default 640)\n"
        << "  --height H       Requested camera height (default 480)\n"
        << "  --warmup N       Warm-up frames to discard (default 10)\n"
        << "  --frames N       Measured frames (default 100)\n"
        << "  --threads N      OpenCV thread count (default 1)\n"
        << "  --csv FILE       Raw timing CSV (default timings.csv)\n"
        << "  --no-display     Disable imshow/waitKey\n"
        << "  --help           Show this message\n\n"
        << "Examples:\n"
        << "  " << program << " 3\n"
        << "  " << program << " --kernel 5 --video video.mp4 --frames 100 --csv vga_n5.csv\n";
}

int parse_positive_int(const std::string& value, const std::string& name)
{
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed <= 0 ||
        parsed > 1000000L) {
        throw std::invalid_argument("Invalid value for " + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char* argv[])
{
    Options options;
    int i = 1;

    // Preserve the assignment's original invocation: ./projet 3
    if (i < argc && argv[i][0] != '-') {
        options.kernel = parse_positive_int(argv[i], "kernel");
        ++i;
    }

    while (i < argc) {
        const std::string arg = argv[i++];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--no-display") {
            options.display = false;
        } else if (arg == "--kernel" || arg == "--width" || arg == "--height" ||
                   arg == "--warmup" || arg == "--frames" || arg == "--threads" ||
                   arg == "--camera" || arg == "--video" || arg == "--csv") {
            if (i >= argc) {
                throw std::invalid_argument("Missing value after " + arg);
            }
            const std::string value = argv[i++];
            if (arg == "--kernel") options.kernel = parse_positive_int(value, arg);
            else if (arg == "--width") options.width = parse_positive_int(value, arg);
            else if (arg == "--height") options.height = parse_positive_int(value, arg);
            else if (arg == "--warmup") {
                char* end = nullptr;
                const long v = std::strtol(value.c_str(), &end, 10);
                if (end == value.c_str() || *end != '\0' || v < 0 || v > 1000000L)
                    throw std::invalid_argument("Invalid value for " + arg + ": " + value);
                options.warmup = static_cast<int>(v);
            }
            else if (arg == "--frames") options.frames = parse_positive_int(value, arg);
            else if (arg == "--threads") options.opencv_threads = parse_positive_int(value, arg);
            else if (arg == "--camera") {
                char* end = nullptr;
                const long v = std::strtol(value.c_str(), &end, 10);
                if (end == value.c_str() || *end != '\0' || v < 0 || v > 1000000L)
                    throw std::invalid_argument("Invalid camera index: " + value);
                options.camera_index = static_cast<int>(v);
            }
            else if (arg == "--video") {
                options.use_video = true;
                options.video_path = value;
            }
            else if (arg == "--csv") options.csv_path = value;
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.kernel < 3 || (options.kernel % 2) == 0) {
        throw std::invalid_argument("Kernel size must be odd and >= 3");
    }
    if (options.frames < 100) {
        std::cerr << "Warning: project protocol requires at least 100 measured frames.\n";
    }
    return options;
}

Difference compare_images(const cv::Mat& reference, const cv::Mat& result)
{
    if (reference.size() != result.size() || reference.type() != result.type()) {
        throw std::runtime_error("Cannot compare images with different sizes or types");
    }

    cv::Mat difference;
    cv::absdiff(reference, result, difference);

    Difference stats;
    stats.differing_pixels = cv::countNonZero(difference);
    double min_value = 0.0;
    cv::minMaxLoc(difference, &min_value, &stats.max_absolute_difference);
    return stats;
}

double percentile_sorted(const std::vector<double>& sorted_values, double fraction)
{
    if (sorted_values.empty()) return 0.0;
    const double position = fraction * static_cast<double>(sorted_values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight;
}

void print_statistics(const std::string& name, std::vector<double> values)
{
    if (values.empty()) {
        std::cout << name << ": no measurements\n";
        return;
    }
    std::sort(values.begin(), values.end());
    const double median = percentile_sorted(values, 0.5);
    const double q1 = percentile_sorted(values, 0.25);
    const double q3 = percentile_sorted(values, 0.75);
    const double minimum = values.front();
    const double maximum = values.back();

    std::cout << std::fixed << std::setprecision(4)
              << name << " (ms): median=" << median
              << ", min=" << minimum
              << ", max=" << maximum
              << ", spread(max-min)=" << (maximum - minimum)
              << ", IQR=" << (q3 - q1) << "\n";
}

cv::Mat to_grayscale(const cv::Mat& frame)
{
    cv::Mat gray;
    if (frame.channels() == 1) {
        gray = frame;
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported number of input image channels");
    }
    return gray;
}

bool read_next_frame(cv::VideoCapture& capture, bool from_video, cv::Mat& frame)
{
    if (capture.read(frame) && !frame.empty()) return true;

    // For a recorded video, loop back to its first frame so that the requested
    // number of measurements can be collected from a short deterministic clip.
    if (from_video) {
        capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
        return capture.read(frame) && !frame.empty();
    }
    return false;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const Options options = parse_options(argc, argv);
        cv::setNumThreads(options.opencv_threads);

        cv::VideoCapture capture;
        if (options.use_video) capture.open(options.video_path);
        else capture.open(options.camera_index);

        if (!capture.isOpened()) {
            std::cerr << "Error: could not open "
                      << (options.use_video ? options.video_path : "camera") << "\n";
            return 1;
        }

        if (!options.use_video) {
            capture.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
        }

        std::ofstream csv(options.csv_path.c_str());
        if (!csv) {
            std::cerr << "Error: could not open CSV file: " << options.csv_path << "\n";
            return 1;
        }
        csv << "measured_frame,my_median_ms,my_sobel_ms,pipeline_ms,"
               "opencv_median_ms,opencv_sobel_ms,"
               "median_differing_pixels,median_max_abs_diff,"
               "sobel_differing_pixels,sobel_max_abs_diff\n";

        timespec timer_resolution{};
        if (clock_getres(CLOCK_MONOTONIC, &timer_resolution) == 0) {
            const long long resolution_ns =
                static_cast<long long>(timer_resolution.tv_sec) * 1000000000LL +
                timer_resolution.tv_nsec;
            std::cout << "CLOCK_MONOTONIC resolution: " << resolution_ns
                      << " ns\n";
        } else {
            std::cerr << "Warning: clock_getres(CLOCK_MONOTONIC) failed\n";
        }

        std::cout << "OpenCV threads: " << cv::getNumThreads() << "\n"
                  << "Kernel: " << options.kernel << "x" << options.kernel << "\n"
                  << "Warm-up frames: " << options.warmup << "\n"
                  << "Measured frames requested: " << options.frames << "\n"
                  << "Timer: clock_gettime(CLOCK_MONOTONIC)\n"
                  << "CSV: " << options.csv_path << "\n";

        std::vector<double> median_custom_times;
        std::vector<double> sobel_custom_times;
        std::vector<double> pipeline_times;
        std::vector<double> median_reference_times;
        std::vector<double> sobel_reference_times;

        cv::Mat color_frame;
        cv::Mat gray;
        cv::Mat custom_median;
        cv::Mat custom_sobel;
        cv::Mat reference_median;
        cv::Mat reference_sobel;
        cv::Mat gx, gy, abs_gx, abs_gy;

        int measured = 0;
        const int total_iterations = options.warmup + options.frames;

        for (int iteration = 0; iteration < total_iterations; ++iteration) {
            const long long pipeline_start = monotonic_ns();

            if (!read_next_frame(capture, options.use_video, color_frame)) {
                std::cerr << "Input ended before all requested frames were processed.\n";
                break;
            }

            gray = to_grayscale(color_frame);

            const long long median_start = monotonic_ns();
            my_median(gray, custom_median, options.kernel);
            const long long median_end = monotonic_ns();

            const long long sobel_start = monotonic_ns();
            my_sobel(custom_median, custom_sobel);
            const long long sobel_end = monotonic_ns();

            if (options.display) {
                cv::imshow("Video input", color_frame);
                cv::imshow("Video gray levels", gray);
                cv::imshow("Video Median - custom", custom_median);
                cv::imshow("Video Sobel - custom", custom_sobel);
                const int key = cv::waitKey(1);
                if (key == 27) break;
            }

            const long long pipeline_end = monotonic_ns();

            // The first 'warmup' iterations are executed but not recorded.
            if (iteration < options.warmup) continue;

            // OpenCV reference is measured outside the custom-pipeline timer.
            const long long ref_median_start = monotonic_ns();
            cv::medianBlur(gray, reference_median, options.kernel);
            const long long ref_median_end = monotonic_ns();

            const long long ref_sobel_start = monotonic_ns();
            cv::Sobel(reference_median, gx, CV_16S, 1, 0, 3, 1.0, 0.0,
                     cv::BORDER_REPLICATE);
            cv::Sobel(reference_median, gy, CV_16S, 0, 1, 3, 1.0, 0.0,
                     cv::BORDER_REPLICATE);
            cv::convertScaleAbs(gx, abs_gx);
            cv::convertScaleAbs(gy, abs_gy);
            cv::addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0.0, reference_sobel);
            const long long ref_sobel_end = monotonic_ns();

            const Difference median_difference =
                compare_images(reference_median, custom_median);
            const Difference sobel_difference =
                compare_images(reference_sobel, custom_sobel);

            FrameTiming timing;
            timing.my_median_ms = elapsed_ms(median_start, median_end);
            timing.my_sobel_ms = elapsed_ms(sobel_start, sobel_end);
            timing.pipeline_ms = elapsed_ms(pipeline_start, pipeline_end);
            timing.ref_median_ms = elapsed_ms(ref_median_start, ref_median_end);
            timing.ref_sobel_ms = elapsed_ms(ref_sobel_start, ref_sobel_end);
            median_custom_times.push_back(timing.my_median_ms);
            sobel_custom_times.push_back(timing.my_sobel_ms);
            pipeline_times.push_back(timing.pipeline_ms);
            median_reference_times.push_back(timing.ref_median_ms);
            sobel_reference_times.push_back(timing.ref_sobel_ms);

            csv << measured << ','
                << std::setprecision(9) << timing.my_median_ms << ','
                << timing.my_sobel_ms << ','
                << timing.pipeline_ms << ','
                << timing.ref_median_ms << ','
                << timing.ref_sobel_ms << ','
                << median_difference.differing_pixels << ','
                << median_difference.max_absolute_difference << ','
                << sobel_difference.differing_pixels << ','
                << sobel_difference.max_absolute_difference << '\n';

            if (measured == 0) {
                std::cout << "Actual frame resolution: " << gray.cols << "x" << gray.rows
                          << " (" << gray.total() << " pixels)\n";
            }

            if (median_difference.differing_pixels != 0 ||
                median_difference.max_absolute_difference != 0.0) {
                std::cerr << "WARNING: median output differs from OpenCV on measured frame "
                          << measured << "\n";
            }
            if (sobel_difference.differing_pixels != 0) {
                std::cerr << "WARNING: Sobel output differs from OpenCV on measured frame "
                          << measured << " (max abs diff="
                          << sobel_difference.max_absolute_difference << ")\n";
            }

            ++measured;
        }

        csv.close();

        if (measured == 0) {
            std::cerr << "No measured frames were collected.\n";
            return 1;
        }

        std::cout << "\n========== CUSTOM BASELINE STATISTICS ==========\n";
        print_statistics("Custom median", median_custom_times);
        print_statistics("Custom Sobel", sobel_custom_times);
        print_statistics("Custom pipeline (capture + grayscale + filters + display if enabled)",
                          pipeline_times);

        std::cout << "\n========== OPENCV REFERENCE STATISTICS ==========\n";
        print_statistics("OpenCV median", median_reference_times);
        print_statistics("OpenCV Sobel", sobel_reference_times);

        std::vector<double> sorted_custom_median = median_custom_times;
        std::vector<double> sorted_reference_median = median_reference_times;
        std::vector<double> sorted_custom_sobel = sobel_custom_times;
        std::vector<double> sorted_reference_sobel = sobel_reference_times;
        std::sort(sorted_custom_median.begin(), sorted_custom_median.end());
        std::sort(sorted_reference_median.begin(), sorted_reference_median.end());
        std::sort(sorted_custom_sobel.begin(), sorted_custom_sobel.end());
        std::sort(sorted_reference_sobel.begin(), sorted_reference_sobel.end());

        const double median_custom = percentile_sorted(sorted_custom_median, 0.5);
        const double median_reference = percentile_sorted(sorted_reference_median, 0.5);
        const double sobel_custom = percentile_sorted(sorted_custom_sobel, 0.5);
        const double sobel_reference = percentile_sorted(sorted_reference_sobel, 0.5);

        std::cout << std::fixed << std::setprecision(3)
                  << "\nMedian speedup (OpenCV median / custom median): "
                  << (median_custom > 0.0 ? median_reference / median_custom : 0.0) << "x\n"
                  << "Sobel speedup (OpenCV Sobel / custom Sobel): "
                  << (sobel_custom > 0.0 ? sobel_reference / sobel_custom : 0.0) << "x\n"
                  << "Measured frames: " << measured << "\n"
                  << "Raw CSV saved to: " << options.csv_path << "\n";

        return 0;
    }
    catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << "\n";
        return 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
