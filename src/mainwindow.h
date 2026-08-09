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

    // Komut satirindan --video ile verilen dosyayi acar. Gosterim ve demo
    // kaydi icin: uygulama acilir acilmaz isleme baslasin.
    void openVideo(const QString& path);

    // --history bayragi arayuz modunda da gecerli olsun diye. Kutuyu isaretlemek
    // zaten onControlsChanged'i tetikliyor, ayrica ayar gondermeye gerek yok.
    void setMotionHistoryEnabled(bool enabled);

signals:
    // Bu sinyaller isci ipligindeki VideoWorker slotlarina bagli. Iplikler
    // farkli oldugu icin Qt baglantiyi otomatik olarak kuyruklu yapar.
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
    void onFrameReady(const QImage& original, const QImage& processed,
                      FrameStats stats, double fps);
    void onSourceOpened(const QString& description, int width, int height, double sourceFps);
    void onSourceClosed();
    void onError(const QString& message);

private:
    QWidget* buildControlPanel();
    PipelineConfig currentConfig() const;
    void updateImageLabels();

    // --- isci iplik ---
    QThread* thread_ = nullptr;
    VideoWorker* worker_ = nullptr;

    // --- goruntu alanlari ---
    QLabel* originalLabel_ = nullptr;
    QLabel* processedLabel_ = nullptr;
    QImage lastOriginal_;
    QImage lastProcessed_;

    // --- kaynak kontrolleri ---
    QSpinBox* cameraIndexSpin_ = nullptr;
    QPushButton* startCameraButton_ = nullptr;
    QPushButton* openFileButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;

    // --- hat kontrolleri ---
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

    // --- durum cubugu ---
    QLabel* fpsStatus_ = nullptr;
    QLabel* timingStatus_ = nullptr;
    QLabel* detectionStatus_ = nullptr;
    QLabel* sourceStatus_ = nullptr;
};
