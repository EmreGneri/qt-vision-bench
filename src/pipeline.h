#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

#include <vector>

// ============================================================================
// AYARLAR (arayuzden de degistirilebilir, varsayilanlar burada)
// ============================================================================
struct PipelineConfig {
    bool   useGrayscale     = true;   // renk bilgisi hareket tespitinde gerekmiyor, atinca hizlaniyor
    bool   useBlur          = true;   // sensor gurultusunu bastirir, sahte konturlari azaltir
    int    blurKernel       = 5;      // tek sayi olmali; process() icinde zorlaniyor
    bool   useCanny         = false;  // kenar goruntusu; tespitten bagimsiz, sadece gorsel cikti
    double cannyLow         = 50.0;
    double cannyHigh        = 150.0;
    bool   useMotionDetect  = true;   // MOG2 arka plan cikarma + kontur
    int    minContourArea   = 500;    // bu alanin altindaki hareket gurultu sayilir
    double mog2VarThreshold = 16.0;   // OpenCV varsayilani
    int    mog2History      = 500;    // arka plan modelinin hafiza uzunlugu (kare)
    bool   drawBoxes        = true;   // tespit kutularini ciktiya ciz

    // reset() sonrasi ilk karelerde MOG2'nin modeli bos: tum kare "hareket"
    // gorunur ve ekrani saran sahte bir kutu cizilir. Bu kadar kare boyunca
    // model beslenir ama tespit bildirilmez.
    int    warmupFrames     = 5;

    // Hareket izi: her karede eski iz sonumlenir ve yeni maskeyle birlestirilir,
    // boylece son saniyelerdeki hareketin yolu goruntude kaliyor.
    //
    // Bu adimin tek bir OpenCV cagrisi karsiligi yok; el yazimi piksel dongusu.
    // Karsilastirmada bilerek var: hattin geri kalani kutuphane cagrilarindan
    // ibaret oldugu icin dil farkini gostermiyordu.
    bool   useMotionHistory = false;
    int    historyDecay     = 240;  // (deger * decay) >> 8, yani ~0.94 sonum
};

// Tek bir karenin islenme suresi. Milisaniye cinsinden, steady_clock ile olculur.
struct FrameStats {
    double totalMs      = 0.0;
    double preprocessMs = 0.0;  // gri tonlama + bulanik + kenar
    double detectMs     = 0.0;  // arka plan cikarma + kontur + filtre
    double historyMs    = 0.0;  // hareket izi (kapaliysa 0)
    int    detections   = 0;
};

// ============================================================================
// Pipeline
//
// Tasarim notu: ara matrisler (gray_, blurred_ ...) uye degisken olarak
// tutuluyor. Boylece her karede yeniden bellek ayrilmiyor; OpenCV ayni tamponu
// yeniden kullaniyor. Python tarafinda bu kontrol yok, olcumdeki farkin bir
// kismi tam olarak buradan geliyor.
// ============================================================================
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg = PipelineConfig{});

    void setConfig(const PipelineConfig& cfg);
    const PipelineConfig& config() const { return cfg_; }

    // Arka plan modelini sifirlar. Kaynak degistiginde cagrilmali, yoksa
    // onceki videonun arka plani yeni karelerde sahte hareket uretir.
    void reset();

    // input: BGR veya tek kanal. output: her zaman BGR (cizime uygun).
    // Bos girdi guvenli: bos cikti ve sifir istatistik doner.
    FrameStats process(const cv::Mat& input, cv::Mat& output);

    const std::vector<cv::Rect>& lastDetections() const { return detections_; }

    // Hareket izi goruntusu (CV_8UC1). Adim kapaliysa bos.
    const cv::Mat& motionHistory() const { return motionHistory_; }

private:
    void ensureBackgroundSubtractor();
    void updateMotionHistory(const cv::Mat& mask);

    PipelineConfig cfg_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSub_;
    std::vector<cv::Rect> detections_;
    int framesSinceReset_ = 0;

    // yeniden kullanilan ara tamponlar
    cv::Mat gray_, blurred_, edges_, mask_, morphKernel_, motionHistory_;
};

// Cift sayi veya sifir gelirse GaussianBlur patlar; disaridan gelen degeri
// tek sayiya ve alt sinira zorlar. Arayuzdeki slider da bunu kullaniyor.
int sanitizeOddKernel(int k);
