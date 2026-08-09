#include "worker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QTimer>

#include <opencv2/imgproc.hpp>

namespace {

// Weight of the newest sample in the FPS average. Lower means a steadier number.
constexpr double kFpsSmoothing = 0.15;

QElapsedTimer& monotonicTimer() {
    static QElapsedTimer timer;
    if (!timer.isValid()) {
        timer.start();
    }
    return timer;
}

}  // namespace

// .copy() is required: the QImage below wraps the buffer's pixels, and that
// buffer is reused on the next frame. Without a deep copy the UI would show
// whatever the next conversion writes over it.
QImage VideoWorker::matToQImage(const cv::Mat& mat, cv::Mat& buffer) {
    if (mat.empty()) {
        return QImage();
    }

    switch (mat.channels()) {
        case 1: cv::cvtColor(mat, buffer, cv::COLOR_GRAY2RGB); break;
        case 3: cv::cvtColor(mat, buffer, cv::COLOR_BGR2RGB); break;
        case 4: cv::cvtColor(mat, buffer, cv::COLOR_BGRA2RGB); break;
        default: return QImage();
    }

    return QImage(buffer.data, buffer.cols, buffer.rows, static_cast<int>(buffer.step),
                  QImage::Format_RGB888)
        .copy();
}

VideoWorker::VideoWorker(QObject* parent) : QObject(parent) {
    timer_ = new QTimer(this);
    // Interval 0: fires as soon as the event loop is idle. On a file source that
    // means "as fast as possible"; on a camera, read() already paces us.
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
    // Rounded, not truncated: 1000/240 would truncate to 4 ms, i.e. 250 FPS.
    const int interval = (maxFps_ > 0) ? static_cast<int>(1000.0 / maxFps_ + 0.5) : 0;
    timer_->setInterval(interval);
    if (cap_.isOpened() && !timer_->isActive()) {
        timer_->start();
    }
}

void VideoWorker::emitOpened(const QString& description) {
    const int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap_.get(cv::CAP_PROP_FPS);

    pipeline_.reset();  // new source means a new background model
    smoothedFps_ = 0.0;
    lastFrameNs_ = 0;

    emit sourceOpened(description, width, height, fps);
    timer_->start();  // the interval is already set by the last setMaxFps()
}

void VideoWorker::openCamera(int index) {
    stop();

#ifdef _WIN32
    // CAP_DSHOW: on Windows the default MSMF backend stalls for 3-5 seconds on
    // some cameras, DirectShow opens immediately.
    const bool opened = cap_.open(index, cv::CAP_DSHOW);
#else
    const bool opened = cap_.open(index);
#endif

    if (!opened) {
        emit errorOccurred(
            QStringLiteral("Cannot open camera %1.\nAnother application may be "
                           "using it, or no camera exists at this index.")
                .arg(index));
        return;
    }

    emitOpened(QStringLiteral("camera %1").arg(index));
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
        // End of file, or the camera dropped out; both count as a normal end.
        stop();
        return;
    }

    const FrameStats stats = pipeline_.process(frame_, processed_);

    // Real FPS: wall-clock distance between frames, not processing time. Source
    // waits and UI cost are included, so it is the number the user experiences.
    const qint64 nowNs = monotonicTimer().nsecsElapsed();
    if (lastFrameNs_ > 0) {
        const double deltaSec = static_cast<double>(nowNs - lastFrameNs_) / 1e9;
        if (deltaSec > 0.0) {
            const double instantFps = 1.0 / deltaSec;
            smoothedFps_ =
                (smoothedFps_ <= 0.0)
                    ? instantFps
                    : (kFpsSmoothing * instantFps + (1.0 - kFpsSmoothing) * smoothedFps_);
        }
    }
    lastFrameNs_ = nowNs;

    emit frameReady(matToQImage(frame_, rgbOriginal_),
                    matToQImage(processed_, rgbProcessed_), stats, smoothedFps_);
}
