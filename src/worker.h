#pragma once

#include <QImage>
#include <QObject>
#include <QString>

#include <opencv2/videoio.hpp>

#include "pipeline.h"

class QTimer;

// ============================================================================
// VideoWorker
//
// Capture and processing do not run on the UI thread: at 30 FPS even 20 ms per
// frame makes the interface feel frozen. This object is moved to its own
// QThread.
//
// Why a QTimer instead of a loop: a while(true) inside a slot blocks that
// thread's event loop, so queued signals such as stop() and applyConfig() would
// never arrive. With a timer the event loop breathes between frames, which also
// means the configuration can be updated safely without a mutex anywhere.
// ============================================================================
class VideoWorker : public QObject {
    Q_OBJECT

public:
    explicit VideoWorker(QObject* parent = nullptr);
    ~VideoWorker() override;

public slots:
    void openCamera(int index);
    void openFile(const QString& path);
    void stop();
    void applyConfig(const PipelineConfig& cfg);
    void setMaxFps(int fps);  // 0 = unlimited (benchmark), >0 = playback rate

signals:
    void frameReady(const QImage& original, const QImage& processed, FrameStats stats,
                    double fps);
    void sourceOpened(const QString& description, int width, int height,
                      double sourceFps);
    void sourceClosed();
    void errorOccurred(const QString& message);

private slots:
    void grabFrame();

private:
    void restartTimer();
    void emitOpened(const QString& description);

    // Converts a BGR cv::Mat into an RGB888 QImage, reusing `buffer` for the
    // colour conversion. The returned QImage owns its pixels.
    static QImage matToQImage(const cv::Mat& mat, cv::Mat& buffer);

    cv::VideoCapture cap_;
    Pipeline pipeline_;
    QTimer* timer_ = nullptr;
    int maxFps_ = 0;

    cv::Mat frame_, processed_;

    // Reused conversion buffers, one per image, so the display path does not
    // allocate a full frame twice per frame either.
    cv::Mat rgbOriginal_, rgbProcessed_;

    // FPS measurement: a single frame delta is far too jumpy, an exponential
    // moving average gives a number that can actually be read.
    double smoothedFps_ = 0.0;
    qint64 lastFrameNs_ = 0;
};
