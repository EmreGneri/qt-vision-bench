#pragma once

#include <QImage>
#include <QMainWindow>

#include "pipeline.h"

class QCheckBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;
class QThread;
class VideoWorker;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // Opens the file passed with --video. For demos and screen recordings:
    // processing starts the moment the window appears.
    void openVideo(const QString& path);

    // Makes the --history flag apply to GUI mode too. Ticking the box already
    // triggers onControlsChanged(), so no separate config push is needed.
    void setMotionHistoryEnabled(bool enabled);

signals:
    // These are connected to VideoWorker slots living on the worker thread.
    // Because the threads differ, Qt makes the connections queued automatically.
    void requestCamera(int index);
    void requestFile(const QString& path);
    void requestStop();
    void requestConfig(const PipelineConfig& cfg);
    void requestMaxFps(int fps);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onStartCamera();
    void onOpenFile();
    void onStop();
    void onControlsChanged();
    void onFrameReady(const QImage& original, const QImage& processed, FrameStats stats,
                      double fps);
    void onSourceOpened(const QString& description, int width, int height,
                        double sourceFps);
    void onSourceClosed();
    void onError(const QString& message);

private:
    QWidget* buildControlPanel();
    PipelineConfig currentConfig() const;
    void updateImageLabels();

    // --- worker thread ---
    QThread* thread_ = nullptr;
    VideoWorker* worker_ = nullptr;

    // --- image areas ---
    QLabel* originalLabel_ = nullptr;
    QLabel* processedLabel_ = nullptr;
    QImage lastOriginal_;
    QImage lastProcessed_;

    // --- source controls ---
    QSpinBox* cameraIndexSpin_ = nullptr;
    QPushButton* startCameraButton_ = nullptr;
    QPushButton* openFileButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;

    // --- pipeline controls ---
    QCheckBox* grayscaleCheck_ = nullptr;
    QCheckBox* blurCheck_ = nullptr;
    QCheckBox* cannyCheck_ = nullptr;
    QCheckBox* motionCheck_ = nullptr;
    QCheckBox* boxesCheck_ = nullptr;
    QCheckBox* historyCheck_ = nullptr;
    QSlider* blurKernelSlider_ = nullptr;
    QLabel* blurKernelValue_ = nullptr;
    QSlider* cannyLowSlider_ = nullptr;
    QSlider* cannyHighSlider_ = nullptr;
    QLabel* cannyValue_ = nullptr;
    QSpinBox* minAreaSpin_ = nullptr;
    QSpinBox* maxFpsSpin_ = nullptr;

    // --- status bar ---
    QLabel* fpsStatus_ = nullptr;
    QLabel* timingStatus_ = nullptr;
    QLabel* detectionStatus_ = nullptr;
    QLabel* sourceStatus_ = nullptr;
};
