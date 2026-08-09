#include "worker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QTimer>

#include <opencv2/imgproc.hpp>

namespace {

// FPS ortalamasinda yeni olcumun agirligi. Dusuk deger = daha durgun sayi.
constexpr double kFpsSmoothing = 0.15;

// cv::Mat BGR -> QImage RGB888.
// .copy() sart: QImage asagidaki gecici cv::Mat'in tamponunu sarmalar, o mat
// fonksiyondan cikinca serbest kalir. Derin kopya olmadan arayuz cop veri gosterir.
QImage matToQImage(const cv::Mat& mat) {
    if (mat.empty()) {
        return QImage();
    }

    cv::Mat rgb;
    switch (mat.channels()) {
        case 1: cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB); break;
        case 3: cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);  break;
        case 4: cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB); break;
        default: return QImage();
    }

    return QImage(rgb.data, rgb.cols, rgb.rows,
                  static_cast<int>(rgb.step), QImage::Format_RGB888)
        .copy();
}

QElapsedTimer& monotonicTimer() {
    static QElapsedTimer timer;
    if (!timer.isValid()) {
        timer.start();
    }
    return timer;
}

}  // namespace

VideoWorker::VideoWorker(QObject* parent) : QObject(parent) {
    timer_ = new QTimer(this);
    // 0 aralik: olay dongusu bosaldiginda hemen tetiklenir. Dosya kaynaginda
    // "olabildigince hizli" demek, kamera kaynaginda read() zaten bekletiyor.
    timer_->setInterval(0);
    timer_->setTimerType(Qt::PreciseTimer);
    connect(timer_, &QTimer::timeout, this, &VideoWorker::grabFrame);
}

VideoWorker::~VideoWorker() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

void VideoWorker::restartTimer() {
    const int interval = (maxFps_ > 0) ? (1000 / maxFps_) : 0;
    timer_->setInterval(interval);
    if (cap_.isOpened() && !timer_->isActive()) {
        timer_->start();
    }
}

void VideoWorker::emitOpened(const QString& description) {
    const int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap_.get(cv::CAP_PROP_FPS);

    pipeline_.reset();  // yeni kaynak = yeni arka plan modeli
    smoothedFps_ = 0.0;
    lastFrameNs_ = 0;

    emit sourceOpened(description, width, height, fps);
    timer_->start();
}

void VideoWorker::openCamera(int index) {
    stop();

    // CAP_DSHOW: Windows'ta varsayilan MSMF arka ucu bazi kameralarda
    // acilista 3-5 saniye takiliyor, DirectShow aninda aciliyor.
    if (!cap_.open(index, cv::CAP_DSHOW)) {
        emit errorOccurred(
            QStringLiteral("Cannot open camera %1.\nAnother application may be "
                           "using it, or no camera exists at this index.")
                .arg(index));
        return;
    }

    emitOpened(QStringLiteral("Kamera %1").arg(index));
}

void VideoWorker::openFile(const QString& path) {
    stop();

    if (!QFileInfo::exists(path)) {
        emit errorOccurred(QStringLiteral("File not found: %1").arg(path));
        return;
    }

    if (!cap_.open(path.toStdString())) {
        emit errorOccurred(
            QStringLiteral("Cannot open video: %1\nThe codec may be unsupported "
                           "(try mp4v or H.264).")
                .arg(path));
        return;
    }

    emitOpened(QFileInfo(path).fileName());
}

void VideoWorker::stop() {
    timer_->stop();
    if (cap_.isOpened()) {
        cap_.release();
        emit sourceClosed();
    }
}

void VideoWorker::applyConfig(const PipelineConfig& cfg) {
    pipeline_.setConfig(cfg);
}

void VideoWorker::setMaxFps(int fps) {
    maxFps_ = (fps < 0) ? 0 : fps;
    restartTimer();
}

void VideoWorker::grabFrame() {
    if (!cap_.isOpened()) {
        timer_->stop();
        return;
    }

    if (!cap_.read(frame_) || frame_.empty()) {
        // Dosya bitti ya da kamera koptu; ikisi de normal sonlanma sayilir.
        stop();
        return;
    }

    const FrameStats stats = pipeline_.process(frame_, processed_);

    // Gercek FPS: islem suresi degil, kareler arasi duvar saati farki.
    // Kaynak beklemesi ve arayuz maliyeti dahil, yani kullanicinin gordugu sayi.
    const qint64 nowNs = monotonicTimer().nsecsElapsed();
    if (lastFrameNs_ > 0) {
        const double deltaSec = static_cast<double>(nowNs - lastFrameNs_) / 1e9;
        if (deltaSec > 0.0) {
            const double instantFps = 1.0 / deltaSec;
            smoothedFps_ = (smoothedFps_ <= 0.0)
                               ? instantFps
                               : (kFpsSmoothing * instantFps +
                                  (1.0 - kFpsSmoothing) * smoothedFps_);
        }
    }
    lastFrameNs_ = nowNs;

    emit frameReady(matToQImage(frame_), matToQImage(processed_), stats, smoothedFps_);
}
