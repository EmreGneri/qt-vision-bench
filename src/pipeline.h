#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

#include <vector>

// ============================================================================
// Tunable parameters. The UI edits a copy of this struct and hands it to the
// worker thread; the defaults below are what the benchmark measures.
// ============================================================================
struct PipelineConfig {
    bool useGrayscale = true;  // colour is irrelevant to motion; dropping it is faster
    bool useBlur = true;       // suppresses the sensor noise that becomes false contours
    int blurKernel = 5;        // must be odd; process() enforces that
    bool useCanny = false;     // edge view only, independent of detection
    double cannyLow = 50.0;
    double cannyHigh = 150.0;
    bool useMotionDetect = true;     // MOG2 background subtraction + contours
    int minContourArea = 500;        // motion smaller than this is treated as noise
    double mog2VarThreshold = 16.0;  // OpenCV default
    int mog2History = 500;           // length of the background model's memory, in frames
    bool drawBoxes = true;           // draw detection boxes into the output image

    // Right after reset() the MOG2 model is empty, so the whole frame reads as
    // foreground and a bounding box is drawn around the entire image. For this
    // many frames the model is fed but no detection is reported.
    int warmupFrames = 5;

    // Motion history trail: every frame decays the previous trail and merges the
    // current mask into it, so the path of the last few seconds stays visible.
    //
    // This stage has no single OpenCV equivalent; it is a hand-written per-pixel
    // loop. It exists in the comparison on purpose: the rest of the pipeline is
    // nothing but library calls, which says very little about the language.
    bool useMotionHistory = false;
    int historyDecay = 240;  // (value * decay) >> 8, i.e. ~0.94 decay per frame
};

// Per-frame timings in milliseconds, measured with steady_clock.
struct FrameStats {
    double totalMs = 0.0;
    double preprocessMs = 0.0;  // grayscale + blur + edges
    double detectMs = 0.0;      // background subtraction + contours + area filter
    double historyMs = 0.0;     // motion history trail (0 when the stage is off)
    int detections = 0;
};

// ============================================================================
// Pipeline
//
// Design note: the intermediate matrices (gray_, blurred_, ...) are members, so
// no memory is allocated per frame - OpenCV writes into the same buffers every
// time. The Python side has no such control by default, and a large part of the
// measured gap comes from exactly that. bench/baseline.py --preallocate isolates
// it by passing dst= buffers into every call.
// ============================================================================
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg = PipelineConfig{});

    // Copying would silently share the MOG2 model between two pipelines that
    // then diverge. Nothing needs it, so it is not allowed.
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void setConfig(const PipelineConfig& cfg);
    const PipelineConfig& config() const { return cfg_; }

    // Rebuilds the background model. Must be called when the source changes,
    // otherwise the previous video's background produces phantom motion.
    void reset();

    // input: BGR or single channel. output: always BGR, ready to draw on.
    // An empty input is safe: empty output, zeroed stats.
    FrameStats process(const cv::Mat& input, cv::Mat& output);

    const std::vector<cv::Rect>& lastDetections() const { return detections_; }

    // Motion history image (CV_8UC1). Empty while the stage is off.
    const cv::Mat& motionHistory() const { return motionHistory_; }

private:
    void ensureBackgroundSubtractor();
    void updateMotionHistory(const cv::Mat& mask);

    PipelineConfig cfg_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSub_;
    std::vector<cv::Rect> detections_;
    int framesSinceReset_ = 0;

    // reused intermediate buffers
    cv::Mat gray_, blurred_, edges_, mask_, morphKernel_, motionHistory_;
};

// GaussianBlur throws on an even or zero kernel size; this clamps whatever comes
// from outside to an odd value at or above 1. The UI slider goes through it too.
int sanitizeOddKernel(int k);
