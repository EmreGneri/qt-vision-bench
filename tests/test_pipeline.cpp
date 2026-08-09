// vision_core birim testleri.
//
// Bilerek test cercevesi kullanilmadi: tek bir bagimlilik eklemeden CTest ile
// kosuyor (ctest --output-on-failure). Cerceve, test sayisi buyudugunde eklenir.

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

// Sabit gri arka plan. MOG2'nin varyansi ogrenmesi icin gurultusuz kare yeterli.
cv::Mat backgroundFrame(int width = 320, int height = 240) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(100, 100, 100));
}

// Arka planin uzerine beyaz dikdortgen: "hareket eden nesne".
cv::Mat frameWithObject(const cv::Rect& box) {
    cv::Mat frame = backgroundFrame();
    cv::rectangle(frame, box, cv::Scalar(255, 255, 255), cv::FILLED);
    return frame;
}

void testKernelSanitizer() {
    section("sanitizeOddKernel");
    CHECK(sanitizeOddKernel(5) == 5);
    CHECK(sanitizeOddKernel(4) == 5);   // cift sayi bir buyuge yuvarlanir
    CHECK(sanitizeOddKernel(0) == 1);   // sifir GaussianBlur'u patlatir
    CHECK(sanitizeOddKernel(-7) == 1);  // negatif de guvenli alt sinira ceker
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
    // Cikti her zaman BGR olmali; arayuz ve kutu cizimi buna guveniyor.
    CHECK(output.type() == CV_8UC3);
}

void testCannyWithoutGrayscale() {
    section("canny works with grayscale disabled");
    PipelineConfig cfg;
    cfg.useGrayscale = false;  // Canny tek kanal ister, hat kendi cevirmeli
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

    // Once arka plan ogretilir. Ilk karelerde her sey "yeni" gorunur.
    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }

    const cv::Rect object(120, 80, 120, 100);  // alan 12000 > minContourArea (500)
    const FrameStats stats = pipeline.process(frameWithObject(object), output);

    CHECK(stats.detections >= 1);

    if (!pipeline.lastDetections().empty()) {
        const cv::Rect found = pipeline.lastDetections().front();
        // Bulanik filtre kenarlari birkac piksel tasirir, tam esitlik beklenmez.
        const cv::Rect intersection = found & object;
        const double overlapRatio =
            static_cast<double>(intersection.area()) / static_cast<double>(object.area());
        CHECK(overlapRatio > 0.7);
    }
}

void testMinAreaFilter() {
    section("min contour area filters small motion");
    PipelineConfig cfg;
    cfg.minContourArea = 100000;  // sahnedeki her seyden buyuk
    Pipeline pipeline(cfg);
    cv::Mat output;

    for (int i = 0; i < 60; ++i) {
        pipeline.process(backgroundFrame(), output);
    }
    const FrameStats stats = pipeline.process(frameWithObject(cv::Rect(120, 80, 120, 100)), output);

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

    // reset sonrasi ilk kare yeni arka plan sayilir; onceki modelden kalan
    // tespit tasmamali.
    const FrameStats stats = pipeline.process(backgroundFrame(), output);
    CHECK(stats.detections == 0);
    CHECK(pipeline.lastDetections().empty());
}

void testWarmupSuppressesDetections() {
    section("warmup frames report no detections");
    PipelineConfig cfg;
    cfg.warmupFrames = 5;
    Pipeline pipeline(cfg);
    cv::Mat output;

    // Isinma boyunca sahnede nesne olsa bile tespit bildirilmemeli; model
    // henuz neyin arka plan oldugunu bilmiyor.
    for (int i = 0; i < cfg.warmupFrames; ++i) {
        const FrameStats stats = pipeline.process(frameWithObject(cv::Rect(40, 40, 150, 120)), output);
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

    // Nesnenin merkezindeki iz, maske 255 oldugu icin tepe degerinde olmali
    const cv::Point center(object.x + object.width / 2, object.y + object.height / 2);
    const int peak = pipeline.motionHistory().at<uchar>(center);
    CHECK(peak == 255);

    // Nesne sahneden cikinca iz sonumlenmeli: azalmali ama aninda sifirlanmamali
    pipeline.process(backgroundFrame(), output);
    const int afterOneFrame = pipeline.motionHistory().at<uchar>(center);
    CHECK(afterOneFrame < peak);
    CHECK(afterOneFrame > 0);

    // (255 * 240) >> 8 = 239; sonum formulu birebir bu olmali
    CHECK(afterOneFrame == 239);
}

void testMotionHistoryOffCostsNothing() {
    section("motion history off reports no history time");
    Pipeline pipeline;  // varsayilan: kapali
    cv::Mat output;

    for (int i = 0; i < 20; ++i) {
        const FrameStats stats = pipeline.process(backgroundFrame(), output);
        CHECK(stats.historyMs == 0.0);
    }
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
    // total, alt adimlarin toplamindan kucuk olamaz (cikti cizimi de dahil).
    CHECK(stats.totalMs + 1e-9 >= stats.preprocessMs);
}

}  // namespace

int main() {
    testKernelSanitizer();
    testEmptyInput();
    testOutputShape();
    testCannyWithoutGrayscale();
    testMotionDetection();
    testMinAreaFilter();
    testResetClearsBackground();
    testWarmupSuppressesDetections();
    testMotionHistoryDecays();
    testMotionHistoryOffCostsNothing();
    testStatsAreConsistent();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return (g_failures == 0) ? 0 : 1;
}
