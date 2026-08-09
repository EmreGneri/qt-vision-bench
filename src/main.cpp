#include <QApplication>
#include <QMetaType>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mainwindow.h"
#include "pipeline.h"

namespace {

// Derleyici adi: olcumun hangi derleyici ve hangi bayraklarla alindigi
// sonucun parcasi. Farkli derleyicide sayilar degisir.
#if defined(__clang__)
constexpr const char* kCompilerName = "clang " __clang_version__;
#elif defined(__GNUC__)
constexpr const char* kCompilerName = "gcc " __VERSION__;
#elif defined(_MSC_VER)
constexpr const char* kCompilerName = "msvc";
#else
constexpr const char* kCompilerName = "unknown";
#endif

struct BenchOptions {
    std::string videoPath;
    int maxFrames = 0;     // 0 = videonun tamami
    int warmupFrames = 10; // MOG2'nin arka plan modeli oturana kadarki kareler
    std::string jsonOut;   // bos = sadece stdout
};

void printUsage() {
    std::puts(
        "qt_vision_bench\n"
        "\n"
        "  (no arguments)              launch the GUI\n"
        "  --video <path>              launch the GUI and start on this file\n"
        "  --bench <video>             run headless benchmark on a video file\n"
        "  --frames <n>                stop after n measured frames (default: whole video)\n"
        "  --warmup <n>                frames skipped before measuring (default: 10)\n"
        "  --json <path>               also write the result as JSON to <path>\n"
        "  --help                      this text\n");
}

// Yuzdelik: p95 gibi kuyruk degerleri ortalamadan daha bilgilendirici,
// tek bir takilma tum ortalamayi kirletmez ama p95'te gorunur.
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

    Pipeline pipeline;  // varsayilan ayarlar; Python tarafi da ayni varsayilanlari kullaniyor
    cv::Mat frame, processed;

    std::vector<double> totalMs, preMs, detMs;
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

        // Isinma kareleri olcume girmez: ilk karelerde arka plan modeli bos
        // oldugu icin hem sure hem tespit sayisi gercek disi.
        if (seen <= opt.warmupFrames) {
            continue;
        }

        totalMs.push_back(stats.totalMs);
        preMs.push_back(stats.preprocessMs);
        detMs.push_back(stats.detectMs);
        detections += stats.detections;
        ++measured;

        if (opt.maxFrames > 0 && measured >= opt.maxFrames) {
            break;
        }
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const double wallSeconds =
        std::chrono::duration<double>(wallEnd - wallStart).count();

    if (measured == 0) {
        std::fprintf(stderr,
                     "error: no frames measured (video too short for warmup=%d?)\n",
                     opt.warmupFrames);
        return 1;
    }

    const double meanTotal = mean(totalMs);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // processing_fps: sadece isleme hattinin hizi (decode haric).
    // end_to_end_fps: decode + isleme, yani gercek dosya isleme hizi.
    char buffer[2048];
    const int written = std::snprintf(
        buffer, sizeof(buffer),
        "{\n"
        "  \"implementation\": \"cpp\",\n"
        "  \"video\": \"%s\",\n"
        "  \"resolution\": \"%dx%d\",\n"
        "  \"frames_measured\": %d,\n"
        "  \"warmup_frames\": %d,\n"
        "  \"process_ms_mean\": %.4f,\n"
        "  \"process_ms_median\": %.4f,\n"
        "  \"process_ms_p95\": %.4f,\n"
        "  \"process_ms_min\": %.4f,\n"
        "  \"process_ms_max\": %.4f,\n"
        "  \"preprocess_ms_mean\": %.4f,\n"
        "  \"detect_ms_mean\": %.4f,\n"
        "  \"processing_fps\": %.2f,\n"
        "  \"end_to_end_fps\": %.2f,\n"
        "  \"wall_seconds\": %.4f,\n"
        "  \"detections_total\": %lld,\n"
        // Surum bilgisi ciktida durmali: Python tarafi farkli bir OpenCV
        // surumu kullaniyorsa karsilastirmayi okuyan bunu gorebilmeli.
        "  \"opencv_version\": \"%s\",\n"
        "  \"compiler\": \"%s\"\n"
        "}\n",
        opt.videoPath.c_str(), width, height, measured, opt.warmupFrames,
        meanTotal, percentile(totalMs, 50.0), percentile(totalMs, 95.0),
        *std::min_element(totalMs.begin(), totalMs.end()),
        *std::max_element(totalMs.begin(), totalMs.end()),
        mean(preMs), mean(detMs),
        (meanTotal > 0.0) ? (1000.0 / meanTotal) : 0.0,
        (wallSeconds > 0.0) ? (static_cast<double>(measured) / wallSeconds) : 0.0,
        wallSeconds, detections, CV_VERSION, kCompilerName);

    if (written < 0 || written >= static_cast<int>(sizeof(buffer))) {
        std::fprintf(stderr, "error: JSON output truncated\n");
        return 1;
    }

    std::fputs(buffer, stdout);

    if (!opt.jsonOut.empty()) {
        std::FILE* out = std::fopen(opt.jsonOut.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "error: cannot write JSON to %s\n", opt.jsonOut.c_str());
            return 1;
        }
        std::fputs(buffer, out);
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
        const bool hasNext = (i + 1) < argc;

        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        }
        if (arg == "--bench" && hasNext) {
            benchMode = true;
            opt.videoPath = argv[++i];
        } else if (arg == "--frames" && hasNext) {
            opt.maxFrames = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && hasNext) {
            opt.warmupFrames = std::atoi(argv[++i]);
        } else if (arg == "--json" && hasNext) {
            opt.jsonOut = argv[++i];
        } else if (arg == "--video" && hasNext) {
            startupVideo = argv[++i];
        }
    }

    if (benchMode) {
        // Arayuz hic olusturulmuyor: olcume Qt'nin baslangic maliyeti karismasin.
        return runBench(opt);
    }

    QApplication app(argc, argv);

    // Kuyruklu baglantilar bu tipleri Qt meta sistemi uzerinden tasiyor.
    qRegisterMetaType<FrameStats>("FrameStats");
    qRegisterMetaType<PipelineConfig>("PipelineConfig");

    MainWindow window;
    window.show();

    if (!startupVideo.empty()) {
        window.openVideo(QString::fromStdString(startupVideo));
    }

    return app.exec();
}
