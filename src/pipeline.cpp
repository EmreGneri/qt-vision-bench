#include "pipeline.h"

#include <opencv2/imgproc.hpp>

// OpenCV 5 moved the shape-analysis functions (contourArea, boundingRect) out of
// imgproc into a separate "geometry" module. In 4.x the same names live in
// imgproc. This guard keeps the file compiling against both.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <chrono>

namespace {

// steady_clock, not system_clock: a clock adjustment can move system_clock
// backwards and produce negative durations.
using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point& start) {
    const auto delta = Clock::now() - start;
    return std::chrono::duration<double, std::milli>(delta).count();
}

}  // namespace

int sanitizeOddKernel(int k) {
    if (k < 1) {
        return 1;
    }
    // GaussianBlur rejects even kernels; round up to the next odd size
    return (k % 2 == 0) ? k + 1 : k;
}

Pipeline::Pipeline(const PipelineConfig& cfg) : cfg_(cfg) {
    cfg_.blurKernel = sanitizeOddKernel(cfg_.blurKernel);
    // 3x3 elliptical kernel: enough to clear single-pixel noise from the mask,
    // anything larger also erases genuinely small objects.
    morphKernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    ensureBackgroundSubtractor();
}

void Pipeline::ensureBackgroundSubtractor() {
    // detectShadows=false: shadow detection costs roughly 20% per frame and adds
    // nothing to bounding boxes around motion.
    bgSub_ = cv::createBackgroundSubtractorMOG2(cfg_.mog2History, cfg_.mog2VarThreshold,
                                                /*detectShadows=*/false);
    framesSinceReset_ = 0;
}

void Pipeline::setConfig(const PipelineConfig& cfg) {
    // MOG2 fixes its parameters at construction time, so the model only has to
    // be rebuilt when one of those two values actually changed.
    const bool bgParamsChanged = (cfg.mog2History != cfg_.mog2History) ||
                                 (cfg.mog2VarThreshold != cfg_.mog2VarThreshold);
    cfg_ = cfg;
    cfg_.blurKernel = sanitizeOddKernel(cfg_.blurKernel);
    if (bgParamsChanged) {
        ensureBackgroundSubtractor();
    }
}

void Pipeline::reset() {
    ensureBackgroundSubtractor();
    detections_.clear();
    motionHistory_.release();
}

// Single-pass decay and merge.
//
// What it computes: history[p] = max((history[p] * decay) >> 8, mask[p])
//
// This stage is hand-written on purpose. To produce the same result NumPy has to
// traverse the image several times - multiply, shift, maximum, write back - and
// each traversal is a separate pass over every pixel. The loop below does all of
// it in one pass with no temporary image. This is where the language difference
// actually shows up; see the Part 2 table in README.md.
void Pipeline::updateMotionHistory(const cv::Mat& mask) {
    if (motionHistory_.size() != mask.size() || motionHistory_.type() != CV_8UC1) {
        motionHistory_ = cv::Mat::zeros(mask.size(), CV_8UC1);
    }

    const int decay = cfg_.historyDecay;
    const int rows = motionHistory_.rows;
    const int cols = motionHistory_.cols;

    for (int y = 0; y < rows; ++y) {
        uchar* historyRow = motionHistory_.ptr<uchar>(y);
        const uchar* maskRow = mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            const int decayed = (historyRow[x] * decay) >> 8;
            const int merged = (decayed > maskRow[x]) ? decayed : maskRow[x];
            historyRow[x] = static_cast<uchar>(merged);
        }
    }
}

FrameStats Pipeline::process(const cv::Mat& input, cv::Mat& output) {
    FrameStats stats;
    detections_.clear();

    if (input.empty()) {
        output.release();
        return stats;
    }

    const auto tStart = Clock::now();

    // ---- preprocessing ---------------------------------------------------
    // work: the input of the detection and edge stages. A pointer rather than a
    // copy; which buffer it refers to changes from stage to stage.
    const cv::Mat* work = &input;

    if (cfg_.useGrayscale && input.channels() == 3) {
        cv::cvtColor(input, gray_, cv::COLOR_BGR2GRAY);
        work = &gray_;
    } else if (cfg_.useGrayscale && input.channels() == 4) {
        cv::cvtColor(input, gray_, cv::COLOR_BGRA2GRAY);
        work = &gray_;
    }

    if (cfg_.useBlur) {
        const int k = sanitizeOddKernel(cfg_.blurKernel);
        // sigma=0: OpenCV derives it from the kernel size, no extra parameter
        cv::GaussianBlur(*work, blurred_, cv::Size(k, k), 0);
        work = &blurred_;
    }

    const bool wantEdges = cfg_.useCanny;
    if (wantEdges) {
        // Canny needs a single channel; convert here if grayscale is disabled
        if (work->channels() != 1) {
            cv::cvtColor(*work, gray_, cv::COLOR_BGR2GRAY);
            cv::Canny(gray_, edges_, cfg_.cannyLow, cfg_.cannyHigh);
        } else {
            cv::Canny(*work, edges_, cfg_.cannyLow, cfg_.cannyHigh);
        }
    }

    stats.preprocessMs = msSince(tStart);

    // ---- detection -------------------------------------------------------
    const auto tDetect = Clock::now();

    if (cfg_.useMotionDetect) {
        // apply() must run during warm-up as well: the model only settles by
        // being fed frames.
        bgSub_->apply(*work, mask_);

        if (framesSinceReset_ >= cfg_.warmupFrames) {
            // Opening (erode + dilate) removes single-pixel speckle without
            // growing the real blobs. Opening rather than closing, because the
            // problem here is surplus, not holes.
            //
            // In place: the motion-history stage below consumes the same buffer,
            // and bench/baseline.py does the same so both sides feed the trail
            // from an identical mask.
            cv::morphologyEx(mask_, mask_, cv::MORPH_OPEN, morphKernel_);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            detections_.reserve(contours.size());
            for (const auto& contour : contours) {
                if (cv::contourArea(contour) < static_cast<double>(cfg_.minContourArea)) {
                    continue;
                }
                detections_.push_back(cv::boundingRect(contour));
            }
        }

        ++framesSinceReset_;
    }

    stats.detectMs = msSince(tDetect);
    stats.detections = static_cast<int>(detections_.size());

    // ---- motion history --------------------------------------------------
    // Not updated before warm-up ends: the mask marks the whole frame as motion
    // at that point, which would start the trail fully white.
    const bool historyActive = cfg_.useMotionHistory && cfg_.useMotionDetect &&
                               (framesSinceReset_ > cfg_.warmupFrames) && !mask_.empty();
    if (historyActive) {
        const auto tHistory = Clock::now();
        updateMotionHistory(mask_);
        stats.historyMs = msSince(tHistory);
    }

    // ---- output image ----------------------------------------------------
    // Deliberately inside the measured region: this is the frame the UI shows,
    // so its cost belongs in the report. The Python side does the same work.
    if (cfg_.useMotionHistory && cfg_.useMotionDetect && !motionHistory_.empty()) {
        cv::cvtColor(motionHistory_, output, cv::COLOR_GRAY2BGR);
    } else if (wantEdges) {
        cv::cvtColor(edges_, output, cv::COLOR_GRAY2BGR);
    } else if (work->channels() == 1) {
        cv::cvtColor(*work, output, cv::COLOR_GRAY2BGR);
    } else {
        work->copyTo(output);
    }

    if (cfg_.drawBoxes) {
        for (const auto& box : detections_) {
            cv::rectangle(output, box, cv::Scalar(0, 255, 0), 2);
        }
    }

    stats.totalMs = msSince(tStart);
    return stats;
}
