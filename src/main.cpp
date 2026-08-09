#include <QApplication>
#include <QMetaType>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "mainwindow.h"
#include "pipeline.h"

namespace {

// Which compiler produced the binary is part of the result: the same source
// measured with a different compiler gives different numbers.
#if defined(__clang__)
constexpr const char* kCompilerName = "clang " __clang_version__;
#elif defined(__GNUC__)
constexpr const char* kCompilerName = "gcc " __VERSION__;
#elif defined(_MSC_VER)
constexpr const char* kCompilerName = "msvc";
#else
constexpr const char* kCompilerName = "unknown";
#endif

// Set by CMake. A benchmark run from a Debug build is meaningless, so the value
// travels with the result and bench/sweep.py refuses anything but Release.
#ifndef QVB_BUILD_TYPE
#define QVB_BUILD_TYPE "unknown"
#endif

#ifdef NDEBUG
constexpr bool kAssertionsDisabled = true;
#else
constexpr bool kAssertionsDisabled = false;
#endif

struct BenchOptions {
    std::string videoPath;
    int maxFrames = 0;           // 0 = the whole video
    int warmupFrames = 10;       // frames spent letting the MOG2 model settle
    std::string jsonOut;         // empty = stdout only
    std::string historyOut;      // empty = do not dump the trail image
    bool motionHistory = false;  // enable the hand-written per-pixel stage
};

// clang-format off: the help text is aligned by hand, rewrapping it would ruin
// the columns the reader sees on the terminal.
void printUsage() {
    std::puts(
        "qt_vision_bench\n"
        "\n"
        "  (no arguments)              launch the GUI\n"
        "  --video <path>              launch the GUI and start on this file\n"
        "  --bench <video>             run headless benchmark on a video file\n"
        "  --frames <n>                stop after n measured frames (default: all)\n"
        "  --warmup <n>                frames skipped before measuring (default: 10)\n"
        "  --json <path>               also write the result as JSON to <path>\n"
        "  --history                   enable the hand-written motion-history stage\n"
        "                              (applies to both the GUI and --bench)\n"
        "  --dump-history <path.png>   write the final motion-history image; used\n"
        "                              by the parity check (--bench with --history)\n"
        "  --help                      this text\n");
}
// clang-format on

// Returns false on anything that is not a plain integer, so a typo fails loudly
// instead of silently turning into 0.
bool parseInt(const char* text, int& out) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

// Percentiles: tail values such as p95 say more than a mean. One stall does not
// move the mean much, but it is clearly visible at p95.
double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double idx = (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    const auto hi = std::min(lo + 1, values.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

int runBench(const BenchOptions& opt) {
    cv::VideoCapture cap(opt.videoPath);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "error: cannot open video: %s\n", opt.videoPath.c_str());
        return 1;
    }

    // Defaults only; bench/baseline.py starts from the same values
    PipelineConfig config;
    config.useMotionHistory = opt.motionHistory;
    Pipeline pipeline(config);
    cv::Mat frame, processed;

    std::vector<double> totalMs, preMs, detMs, historyMs;
    long long detections = 0;
    int measured = 0;
    int seen = 0;

    const auto wallStart = std::chrono::steady_clock::now();

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        ++seen;

        const FrameStats stats = pipeline.process(frame, processed);

        // Warm-up frames are not measured: while the background model is still
        // empty both the timings and the detection counts are unrepresentative.
        if (seen <= opt.warmupFrames) {
            continue;
        }

        totalMs.push_back(stats.totalMs);
        preMs.push_back(stats.preprocessMs);
        detMs.push_back(stats.detectMs);
        historyMs.push_back(stats.historyMs);
        detections += stats.detections;
        ++measured;

        if (opt.maxFrames > 0 && measured >= opt.maxFrames) {
            break;
        }
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();

    if (measured == 0) {
        std::fprintf(stderr,
                     "error: no frames measured (video too short for warmup=%d?)\n",
                     opt.warmupFrames);
        return 1;
    }

    // The trail image after the last frame. Written as a lossless PNG so the
    // harness can compare it pixel by pixel against the Python one.
    if (!opt.historyOut.empty()) {
        if (pipeline.motionHistory().empty()) {
            std::fprintf(stderr,
                         "error: --dump-history requires --history and a video "
                         "long enough to leave warm-up\n");
            return 1;
        }
        if (!cv::imwrite(opt.historyOut, pipeline.motionHistory())) {
            std::fprintf(stderr, "error: cannot write history image to %s\n",
                         opt.historyOut.c_str());
            return 1;
        }
    }

    const double meanTotal = mean(totalMs);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // processing_fps: the pipeline alone, decode excluded.
    // end_to_end_fps: decode plus processing, i.e. real file throughput.
    std::ostringstream json;
    json << std::fixed;
    json << "{\n"
         << "  \"implementation\": \"cpp\",\n"
         << "  \"video\": \"" << opt.videoPath << "\",\n"
         << "  \"resolution\": \"" << width << "x" << height << "\",\n"
         << "  \"frames_measured\": " << measured << ",\n"
         << "  \"warmup_frames\": " << opt.warmupFrames << ",\n"
         << std::setprecision(4) << "  \"process_ms_mean\": " << meanTotal << ",\n"
         << "  \"process_ms_median\": " << percentile(totalMs, 50.0) << ",\n"
         << "  \"process_ms_p95\": " << percentile(totalMs, 95.0) << ",\n"
         << "  \"process_ms_min\": " << *std::min_element(totalMs.begin(), totalMs.end())
         << ",\n"
         << "  \"process_ms_max\": " << *std::max_element(totalMs.begin(), totalMs.end())
         << ",\n"
         << "  \"preprocess_ms_mean\": " << mean(preMs) << ",\n"
         << "  \"detect_ms_mean\": " << mean(detMs) << ",\n"
         << "  \"history_ms_mean\": " << mean(historyMs) << ",\n"
         << "  \"motion_history\": " << (opt.motionHistory ? "true" : "false") << ",\n"
         << std::setprecision(2)
         << "  \"processing_fps\": " << ((meanTotal > 0.0) ? (1000.0 / meanTotal) : 0.0)
         << ",\n"
         << "  \"end_to_end_fps\": "
         << ((wallSeconds > 0.0) ? (static_cast<double>(measured) / wallSeconds) : 0.0)
         << ",\n"
         << std::setprecision(4) << "  \"wall_seconds\": " << wallSeconds << ",\n"
         << "  \"detections_total\": " << detections
         << ",\n"
         // Version and build information belongs in the output: whoever reads
         // the comparison has to be able to see that the two sides link
         // different OpenCV builds, and that this one was optimised.
         << "  \"opencv_version\": \"" << CV_VERSION << "\",\n"
         << "  \"compiler\": \"" << kCompilerName << "\",\n"
         << "  \"build_type\": \"" << QVB_BUILD_TYPE << "\",\n"
         << "  \"assertions_disabled\": " << (kAssertionsDisabled ? "true" : "false")
         << "\n}\n";

    const std::string text = json.str();
    std::fputs(text.c_str(), stdout);

    if (!opt.jsonOut.empty()) {
        std::FILE* out = std::fopen(opt.jsonOut.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "error: cannot write JSON to %s\n", opt.jsonOut.c_str());
            return 1;
        }
        std::fputs(text.c_str(), out);
        std::fclose(out);
    }

    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    BenchOptions opt;
    bool benchMode = false;
    std::string startupVideo;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        // Every option below needs a value; checking once keeps the branches short.
        const bool takesValue =
            (arg == "--bench" || arg == "--frames" || arg == "--warmup" ||
             arg == "--json" || arg == "--video" || arg == "--dump-history");
        if (takesValue && i + 1 >= argc) {
            std::fprintf(stderr, "error: %s needs a value\n", arg.c_str());
            return 2;
        }

        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "--bench") {
            benchMode = true;
            opt.videoPath = argv[++i];
        } else if (arg == "--frames") {
            if (!parseInt(argv[++i], opt.maxFrames) || opt.maxFrames < 0) {
                std::fprintf(stderr, "error: --frames expects a non-negative integer\n");
                return 2;
            }
        } else if (arg == "--warmup") {
            if (!parseInt(argv[++i], opt.warmupFrames) || opt.warmupFrames < 0) {
                std::fprintf(stderr, "error: --warmup expects a non-negative integer\n");
                return 2;
            }
        } else if (arg == "--json") {
            opt.jsonOut = argv[++i];
        } else if (arg == "--dump-history") {
            opt.historyOut = argv[++i];
        } else if (arg == "--video") {
            startupVideo = argv[++i];
        } else if (arg == "--history") {
            opt.motionHistory = true;
        } else {
            // Silently ignoring a typo would mean reporting a measurement that
            // did not use the flag the caller thought they passed.
            std::fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            printUsage();
            return 2;
        }
    }

    if (!opt.historyOut.empty() && !benchMode) {
        std::fprintf(stderr, "error: --dump-history only applies to --bench\n");
        return 2;
    }

    if (benchMode) {
        // No UI is constructed at all, so Qt's start-up cost stays out of the
        // measurement.
        return runBench(opt);
    }

    QApplication app(argc, argv);

    // Queued connections carry these types through Qt's meta-object system.
    qRegisterMetaType<FrameStats>("FrameStats");
    qRegisterMetaType<PipelineConfig>("PipelineConfig");

    MainWindow window;
    window.setMotionHistoryEnabled(opt.motionHistory);
    window.show();

    if (!startupVideo.empty()) {
        window.openVideo(QString::fromStdString(startupVideo));
    }

    return app.exec();
}
