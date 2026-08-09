// Unit tests for vision_core.
//
// No test framework on purpose: this runs under CTest without adding a single
// dependency (ctest --output-on-failure). A framework goes in when the number of
// tests justifies one.

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <string>

#include "pipeline.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void checkImpl(bool condition, const char* expression, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("  FAIL (line %d): %s\n", line, expression);
    }
}

#define CHECK(cond) checkImpl((cond), #cond, __LINE__)

void section(const std::string& name) {
    std::printf("[ %s ]\n", name.c_str());
}

// Flat grey background. A noise-free frame is enough for MOG2 to learn a variance.
cv::Mat backgroundFrame(int width = 320, int height = 240) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(100, 100, 100));
}

// A white rectangle on that background: the "moving object".
cv::Mat frameWithObject(const cv::Rect& box) {
    cv::Mat frame = backgroundFrame();
    cv::rectangle(frame, box, cv::Scalar(255, 255, 255), cv::FILLED);
    return frame;
}

void testKernelSanitizer() {
    section("sanitizeOddKernel");
    CHECK(sanitizeOddKernel(5) == 5);
    CHECK(sanitizeOddKernel(4) == 5);   // even sizes round up
    CHECK(sanitizeOddKernel(0) == 1);   // zero would throw inside GaussianBlur
    CHECK(sanitizeOddKernel(-7) == 1);  // negatives clamp to the safe minimum
}

void testConfigIsSanitized() {
    section("config kernel is sanitized on both paths");
    PipelineConfig cfg;
    cfg.blurKernel = 8;
    Pipeline pipeline(cfg);
    CHECK(pipeline.config().blurKernel == 9);  // constructor

    PipelineConfig other;
    other.blurKernel = 0;
    pipeline.setConfig(other);
    CHECK(pipeline.config().blurKernel == 1);  // setConfig
}

void testEmptyInput() {
    section("empty input is safe");
    Pipeline pipeline;
    cv::Mat output;
    const FrameStats stats = pipeline.process(cv::Mat(), output);
    CHECK(output.empty());
    CHECK(stats.detections == 0);
    CHECK(stats.totalMs == 0.0);
}

void testOutputShape() {
    section("output shape and type");
    Pipeline pipeline;
    cv::Mat output;
    const cv::Mat input = backgroundFrame(320, 240);
    pipeline.process(input, output);

    CHECK(!output.empty());
    CHECK(output.rows == input.rows);
    CHECK(output.cols == input.cols);
    // The output is always BGR; the UI and the box drawing rely on that.
    CHECK(output.type() == CV_8UC3);
}

void testCannyWithoutGrayscale() {
    section("canny works with grayscale disabled");
    PipelineConfig cfg;
    cfg.useGrayscale = false;  // Canny needs one channel, the pipeline converts
    cfg.useCanny = true;
    Pipeline pipeline(cfg);

    cv::Mat output;
    pipeline.process(frameWithObject(cv::Rect(50, 50, 100, 100)), output);
    CHECK(!output.empty());
    CHECK(output.type() == CV_8UC3);
}

void testMotionDetection() {
    section("motion detection finds a moving object");
    Pipeline pipeline;
    cv::Mat output;

    // Teach the background first; in the opening frames everything looks new.
    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }

    const cv::Rect object(120, 80, 120, 100);  // area 12000 > minContourArea (500)
    const FrameStats stats = pipeline.process(frameWithObject(object), output);

    CHECK(stats.detections >= 1);

    if (!pipeline.lastDetections().empty()) {
        const cv::Rect found = pipeline.lastDetections().front();
        // Blur spreads the edges by a few pixels, so exact equality is not expected.
        const cv::Rect intersection = found & object;
        const double overlapRatio =
            static_cast<double>(intersection.area()) / static_cast<double>(object.area());
        CHECK(overlapRatio > 0.7);
    }
}

void testMinAreaFilter() {
    section("min contour area filters small motion");
    PipelineConfig cfg;
    cfg.minContourArea = 100000;  // larger than anything in the scene
    Pipeline pipeline(cfg);
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }
    const FrameStats stats =
        pipeline.process(frameWithObject(cv::Rect(120, 80, 120, 100)), output);

    CHECK(stats.detections == 0);
}

void testResetClearsBackground() {
    section("reset clears the background model");
    Pipeline pipeline;
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }
    pipeline.reset();

    // After reset the first frame is the new background; nothing from the old
    // model may leak through as a detection.
    const FrameStats stats = pipeline.process(backgroundFrame(), output);
    CHECK(stats.detections == 0);
    CHECK(pipeline.lastDetections().empty());
}

void testMog2ParamChangeRebuildsModel() {
    section("changing MOG2 parameters rebuilds the model");
    Pipeline pipeline;
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }

    // A new varThreshold means a new subtractor, which means warm-up starts
    // over: a moving object must not be reported on the very next frame.
    PipelineConfig cfg = pipeline.config();
    cfg.mog2VarThreshold = 32.0;
    pipeline.setConfig(cfg);

    const FrameStats stats =
        pipeline.process(frameWithObject(cv::Rect(120, 80, 120, 100)), output);
    CHECK(stats.detections == 0);

    // A parameter that does not belong to the model must not reset it.
    PipelineConfig same = pipeline.config();
    same.minContourArea = 123;
    pipeline.setConfig(same);
    CHECK(pipeline.config().mog2VarThreshold == 32.0);
}

void testWarmupSuppressesDetections() {
    section("warmup frames report no detections");
    PipelineConfig cfg;
    cfg.warmupFrames = 5;
    Pipeline pipeline(cfg);
    cv::Mat output;

    // Even with an object in the scene nothing may be reported during warm-up;
    // the model does not know yet what the background is.
    for (int i = 0; i < cfg.warmupFrames; ++i) {
        const FrameStats stats =
            pipeline.process(frameWithObject(cv::Rect(40, 40, 150, 120)), output);
        CHECK(stats.detections == 0);
    }
}

void testMotionHistoryDecays() {
    section("motion history decays and merges");
    PipelineConfig cfg;
    cfg.useMotionHistory = true;
    Pipeline pipeline(cfg);
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }

    const cv::Rect object(120, 80, 120, 100);
    const FrameStats withObject = pipeline.process(frameWithObject(object), output);

    CHECK(withObject.historyMs > 0.0);
    CHECK(!pipeline.motionHistory().empty());
    CHECK(pipeline.motionHistory().type() == CV_8UC1);
    CHECK(output.type() == CV_8UC3);
    CHECK(pipeline.motionHistory().size() == output.size());

    // At the centre of the object the mask is 255, so the trail must peak there
    const cv::Point center(object.x + object.width / 2, object.y + object.height / 2);
    const int peak = pipeline.motionHistory().at<uchar>(center);
    CHECK(peak == 255);

    // Once the object leaves, the trail decays: lower, but not instantly zero
    pipeline.process(backgroundFrame(), output);
    const int afterOneFrame = pipeline.motionHistory().at<uchar>(center);
    CHECK(afterOneFrame < peak);
    CHECK(afterOneFrame > 0);

    // (255 * 240) >> 8 = 239; the decay formula must be exactly this, because
    // bench/baseline.py reimplements it and the harness compares the two images
    CHECK(afterOneFrame == 239);

    // ... and it keeps decaying by the same rule: (239 * 240) >> 8 = 224
    pipeline.process(backgroundFrame(), output);
    CHECK(pipeline.motionHistory().at<uchar>(center) == 224);
}

void testMotionHistoryOffCostsNothing() {
    section("motion history off reports no history time");
    Pipeline pipeline;  // default: off
    cv::Mat output;

    for (int i = 0; i < 20; ++i) {
        const FrameStats stats = pipeline.process(backgroundFrame(), output);
        CHECK(stats.historyMs == 0.0);
    }
    CHECK(pipeline.motionHistory().empty());
}

void testResetClearsMotionHistory() {
    section("reset clears the motion history image");
    PipelineConfig cfg;
    cfg.useMotionHistory = true;
    Pipeline pipeline(cfg);
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }
    pipeline.process(frameWithObject(cv::Rect(120, 80, 120, 100)), output);
    CHECK(!pipeline.motionHistory().empty());

    // A new source must not inherit the previous video's trail.
    pipeline.reset();
    CHECK(pipeline.motionHistory().empty());
}

void testStatsAreConsistent() {
    section("timing fields are consistent");
    Pipeline pipeline;
    cv::Mat output;
    const FrameStats stats = pipeline.process(backgroundFrame(), output);

    CHECK(stats.totalMs >= 0.0);
    CHECK(stats.preprocessMs >= 0.0);
    CHECK(stats.detectMs >= 0.0);
    // total cannot be smaller than its parts (it also covers drawing the output).
    CHECK(stats.totalMs + 1e-9 >= stats.preprocessMs);
}

}  // namespace

int main() {
    testKernelSanitizer();
    testConfigIsSanitized();
    testEmptyInput();
    testOutputShape();
    testCannyWithoutGrayscale();
    testMotionDetection();
    testMinAreaFilter();
    testResetClearsBackground();
    testMog2ParamChangeRebuildsModel();
    testWarmupSuppressesDetections();
    testMotionHistoryDecays();
    testMotionHistoryOffCostsNothing();
    testResetClearsMotionHistory();
    testStatsAreConsistent();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
