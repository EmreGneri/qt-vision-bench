#include "pipeline.h"

#include <opencv2/imgproc.hpp>

// OpenCV 5'te sekil analizi fonksiyonlari (contourArea, boundingRect) imgproc'tan
// ayrilip ayri bir "geometry" modulune tasindi. 4.x'te ayni isimler imgproc icinde.
// Bu koruma sayesinde kod iki surumde de derleniyor.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include <chrono>

namespace {

// steady_clock kullaniliyor: system_clock saat ayari degisirse geriye
// gidebilir, olcumde negatif sure uretir.
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
    // GaussianBlur cift cekirdek kabul etmez; bir buyuge yuvarla
    return (k % 2 == 0) ? k + 1 : k;
}

Pipeline::Pipeline(const PipelineConfig& cfg) : cfg_(cfg) {
    // 3x3 eliptik cekirdek: maskedeki tek piksellik gurultuyu temizlemek icin
    // yeterli, daha buyugu gercek kucuk nesneleri de siliyor.
    morphKernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    ensureBackgroundSubtractor();
}

void Pipeline::ensureBackgroundSubtractor() {
    // detectShadows=false: golge tespiti kareyi ~%20 yavaslatiyor ve hareket
    // kutulari icin bir faydasi yok.
    bgSub_ = cv::createBackgroundSubtractorMOG2(
        cfg_.mog2History, cfg_.mog2VarThreshold, /*detectShadows=*/false);
    framesSinceReset_ = 0;
}

void Pipeline::setConfig(const PipelineConfig& cfg) {
    // MOG2 parametreleri nesne olusturulurken sabitleniyor; sadece bunlar
    // degistiyse arka plan modelini bastan kur.
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
}

FrameStats Pipeline::process(const cv::Mat& input, cv::Mat& output) {
    FrameStats stats;
    detections_.clear();

    if (input.empty()) {
        output.release();
        return stats;
    }

    const auto tStart = Clock::now();

    // ---- On isleme -------------------------------------------------------
    // work: tespit ve kenar adimlarinin girdisi. Kopyalama yapmamak icin
    // referans; hangi tampona baktigi adim adim degisiyor.
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
        // sigma=0: OpenCV cekirdek boyutundan hesaplar, ayri parametre gerekmez
        cv::GaussianBlur(*work, blurred_, cv::Size(k, k), 0);
        work = &blurred_;
    }

    const bool wantEdges = cfg_.useCanny;
    if (wantEdges) {
        // Canny tek kanal ister; gri tonlama kapaliysa burada zorunlu olarak yap
        if (work->channels() != 1) {
            cv::cvtColor(*work, gray_, cv::COLOR_BGR2GRAY);
            cv::Canny(gray_, edges_, cfg_.cannyLow, cfg_.cannyHigh);
        } else {
            cv::Canny(*work, edges_, cfg_.cannyLow, cfg_.cannyHigh);
        }
    }

    stats.preprocessMs = msSince(tStart);

    // ---- Tespit ----------------------------------------------------------
    const auto tDetect = Clock::now();

    if (cfg_.useMotionDetect) {
        // apply() isinma sirasinda da cagrilmali: model ancak beslenerek oturuyor.
        bgSub_->apply(*work, mask_);

        if (framesSinceReset_ >= cfg_.warmupFrames) {
            // Acma (erode+dilate): tek piksellik parazitleri siler, gercek
            // bloklari buyutmez. Kapama yerine acma, cunku sorun fazlalik.
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

    // ---- Cikti goruntusu -------------------------------------------------
    // Olcumun disinda tutulmadi: arayuzun gosterdigi kare bu, maliyeti de
    // rapora dahil olmali. Python tarafi da ayni isi yapiyor.
    if (wantEdges) {
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
