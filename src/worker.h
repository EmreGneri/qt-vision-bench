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
// Kare yakalama ve isleme arayuz ipliginde yapilmaz: 30 FPS'te her kare ~20 ms
// surse bile arayuz donuk hisseder. Bu nesne kendi QThread'ine tasinir.
//
// Dongu yerine QTimer kullanilmasinin sebebi: while(true) bir slot icinde
// calisirsa o ipligin olay dongusu bloke olur ve stop()/applyConfig() gibi
// kuyruklanmis sinyaller hic ulasmaz. Timer ile kareler arasinda olay dongusu
// nefes aliyor, ayrica ayarlar mutex'e gerek kalmadan guvenle degisiyor.
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
    void setMaxFps(int fps);  // 0 = sinirsiz (benchmark icin), >0 = oynatma hizi

signals:
    void frameReady(const QImage& original, const QImage& processed,
                    FrameStats stats, double fps);
    void sourceOpened(const QString& description, int width, int height, double sourceFps);
    void sourceClosed();
    void errorOccurred(const QString& message);

private slots:
    void grabFrame();

private:
    void restartTimer();
    void emitOpened(const QString& description);

    cv::VideoCapture cap_;
    Pipeline pipeline_;
    QTimer* timer_ = nullptr;
    int maxFps_ = 0;

    cv::Mat frame_, processed_;

    // FPS olcumu: tek karelik fark cok oynak, ustel hareketli ortalama daha
    // okunabilir bir sayi veriyor.
    double smoothedFps_ = 0.0;
    qint64 lastFrameNs_ = 0;
};
