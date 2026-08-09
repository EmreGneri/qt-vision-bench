#include "mainwindow.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QMetaObject>
#include <QPushButton>
#include <QResizeEvent>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QThread>
#include <QVBoxLayout>

#include "worker.h"

namespace {

// Kontrol panelinin sabit genisligi. Splitter'in solundaki goruntu alani
// kalan tum yeri aliyor.
constexpr int kPanelWidth = 260;

QLabel* makeVideoLabel(const QString& placeholder) {
    auto* label = new QLabel(placeholder);
    label->setAlignment(Qt::AlignCenter);
    label->setMinimumSize(240, 180);
    // Ignored: etiket, icindeki pixmap'in boyutunu layout'a dayatmasin.
    // Aksi halde buyuk kare geldiginde pencere kendi kendine buyur.
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    label->setStyleSheet(QStringLiteral("background-color: #202020; color: #909090;"));
    return label;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QStringLiteral("Qt Vision Bench - C++ / OpenCV"));

    // ---- merkezi alan: iki goruntu + kontrol paneli ----
    originalLabel_ = makeVideoLabel(QStringLiteral("Source\n(no input)"));
    processedLabel_ = makeVideoLabel(QStringLiteral("Processed\n(no input)"));

    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(originalLabel_);
    splitter->addWidget(processedLabel_);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);

    auto* central = new QWidget;
    auto* centralLayout = new QHBoxLayout(central);
    centralLayout->addWidget(splitter, /*stretch=*/1);
    centralLayout->addWidget(buildControlPanel(), /*stretch=*/0);
    setCentralWidget(central);

    // ---- durum cubugu ----
    fpsStatus_ = new QLabel(QStringLiteral("FPS: -"));
    timingStatus_ = new QLabel(QStringLiteral("frame: - ms"));
    detectionStatus_ = new QLabel(QStringLiteral("detections: -"));
    sourceStatus_ = new QLabel(QStringLiteral("source: none"));
    statusBar()->addWidget(sourceStatus_, 1);
    statusBar()->addPermanentWidget(detectionStatus_);
    statusBar()->addPermanentWidget(timingStatus_);
    statusBar()->addPermanentWidget(fpsStatus_);

    // ---- isci iplik ----
    // worker_ bilerek parent'siz olusturuluyor: parent'i olan bir QObject
    // moveToThread ile tasinamaz.
    thread_ = new QThread(this);
    worker_ = new VideoWorker;
    worker_->moveToThread(thread_);
    connect(thread_, &QThread::finished, worker_, &QObject::deleteLater);

    connect(this, &MainWindow::requestCamera, worker_, &VideoWorker::openCamera);
    connect(this, &MainWindow::requestFile, worker_, &VideoWorker::openFile);
    connect(this, &MainWindow::requestStop, worker_, &VideoWorker::stop);
    connect(this, &MainWindow::requestConfig, worker_, &VideoWorker::applyConfig);
    connect(this, &MainWindow::requestMaxFps, worker_, &VideoWorker::setMaxFps);

    connect(worker_, &VideoWorker::frameReady, this, &MainWindow::onFrameReady);
    connect(worker_, &VideoWorker::sourceOpened, this, &MainWindow::onSourceOpened);
    connect(worker_, &VideoWorker::sourceClosed, this, &MainWindow::onSourceClosed);
    connect(worker_, &VideoWorker::errorOccurred, this, &MainWindow::onError);

    thread_->start();

    // Arayuzdeki varsayilan degerleri isciye gonder ki iki taraf ayni ayarla baslasin
    onControlsChanged();

    resize(1100, 680);
}

MainWindow::~MainWindow() {
    if (thread_) {
        // Blocking: yakalama durmadan iplik kapatilirsa VideoCapture yarim kalir.
        QMetaObject::invokeMethod(worker_, "stop", Qt::BlockingQueuedConnection);
        thread_->quit();
        thread_->wait();
    }
}

QWidget* MainWindow::buildControlPanel() {
    auto* panel = new QWidget;
    panel->setFixedWidth(kPanelWidth);
    auto* layout = new QVBoxLayout(panel);

    // ---- kaynak ----
    auto* sourceBox = new QGroupBox(QStringLiteral("Source"));
    auto* sourceLayout = new QVBoxLayout(sourceBox);

    auto* cameraRow = new QHBoxLayout;
    cameraIndexSpin_ = new QSpinBox;
    cameraIndexSpin_->setRange(0, 8);
    cameraIndexSpin_->setPrefix(QStringLiteral("cam "));
    startCameraButton_ = new QPushButton(QStringLiteral("Start"));
    cameraRow->addWidget(cameraIndexSpin_);
    cameraRow->addWidget(startCameraButton_, 1);
    sourceLayout->addLayout(cameraRow);

    openFileButton_ = new QPushButton(QStringLiteral("Open video file..."));
    stopButton_ = new QPushButton(QStringLiteral("Stop"));
    stopButton_->setEnabled(false);
    sourceLayout->addWidget(openFileButton_);
    sourceLayout->addWidget(stopButton_);
    layout->addWidget(sourceBox);

    // ---- hat adimlari ----
    auto* stagesBox = new QGroupBox(QStringLiteral("Pipeline stages"));
    auto* stagesLayout = new QVBoxLayout(stagesBox);
    grayscaleCheck_ = new QCheckBox(QStringLiteral("Grayscale"));
    blurCheck_ = new QCheckBox(QStringLiteral("Gaussian blur"));
    cannyCheck_ = new QCheckBox(QStringLiteral("Canny edges"));
    motionCheck_ = new QCheckBox(QStringLiteral("Motion detection (MOG2)"));
    boxesCheck_ = new QCheckBox(QStringLiteral("Draw bounding boxes"));

    const PipelineConfig defaults;
    grayscaleCheck_->setChecked(defaults.useGrayscale);
    blurCheck_->setChecked(defaults.useBlur);
    cannyCheck_->setChecked(defaults.useCanny);
    motionCheck_->setChecked(defaults.useMotionDetect);
    boxesCheck_->setChecked(defaults.drawBoxes);

    for (QCheckBox* box : {grayscaleCheck_, blurCheck_, cannyCheck_, motionCheck_, boxesCheck_}) {
        stagesLayout->addWidget(box);
        connect(box, &QCheckBox::toggled, this, &MainWindow::onControlsChanged);
    }
    layout->addWidget(stagesBox);

    // ---- parametreler ----
    auto* paramsBox = new QGroupBox(QStringLiteral("Parameters"));
    auto* paramsLayout = new QFormLayout(paramsBox);

    blurKernelSlider_ = new QSlider(Qt::Horizontal);
    blurKernelSlider_->setRange(1, 31);
    blurKernelSlider_->setSingleStep(2);
    blurKernelSlider_->setValue(defaults.blurKernel);
    blurKernelValue_ = new QLabel(QString::number(defaults.blurKernel));
    auto* blurRow = new QHBoxLayout;
    blurRow->addWidget(blurKernelSlider_, 1);
    blurRow->addWidget(blurKernelValue_);
    paramsLayout->addRow(QStringLiteral("Blur kernel"), blurRow);

    cannyLowSlider_ = new QSlider(Qt::Horizontal);
    cannyLowSlider_->setRange(0, 255);
    cannyLowSlider_->setValue(static_cast<int>(defaults.cannyLow));
    cannyHighSlider_ = new QSlider(Qt::Horizontal);
    cannyHighSlider_->setRange(0, 255);
    cannyHighSlider_->setValue(static_cast<int>(defaults.cannyHigh));
    cannyValue_ = new QLabel(QStringLiteral("%1 / %2")
                                 .arg(static_cast<int>(defaults.cannyLow))
                                 .arg(static_cast<int>(defaults.cannyHigh)));
    paramsLayout->addRow(QStringLiteral("Canny low"), cannyLowSlider_);
    paramsLayout->addRow(QStringLiteral("Canny high"), cannyHighSlider_);
    paramsLayout->addRow(QStringLiteral("Thresholds"), cannyValue_);

    minAreaSpin_ = new QSpinBox;
    minAreaSpin_->setRange(0, 50000);
    minAreaSpin_->setSingleStep(100);
    minAreaSpin_->setValue(defaults.minContourArea);
    paramsLayout->addRow(QStringLiteral("Min contour area"), minAreaSpin_);

    maxFpsSpin_ = new QSpinBox;
    maxFpsSpin_->setRange(0, 240);
    maxFpsSpin_->setValue(0);
    // 0 = throttle yok. Dosya kaynaginda "olabildigince hizli" demek, olcum modu.
    maxFpsSpin_->setSpecialValueText(QStringLiteral("unlimited"));
    paramsLayout->addRow(QStringLiteral("Max FPS"), maxFpsSpin_);

    layout->addWidget(paramsBox);
    layout->addStretch(1);

    connect(blurKernelSlider_, &QSlider::valueChanged, this, &MainWindow::onControlsChanged);
    connect(cannyLowSlider_, &QSlider::valueChanged, this, &MainWindow::onControlsChanged);
    connect(cannyHighSlider_, &QSlider::valueChanged, this, &MainWindow::onControlsChanged);
    connect(minAreaSpin_, &QSpinBox::valueChanged, this, &MainWindow::onControlsChanged);
    connect(maxFpsSpin_, &QSpinBox::valueChanged, this, &MainWindow::onControlsChanged);

    connect(startCameraButton_, &QPushButton::clicked, this, &MainWindow::onStartCamera);
    connect(openFileButton_, &QPushButton::clicked, this, &MainWindow::onOpenFile);
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::onStop);

    return panel;
}

PipelineConfig MainWindow::currentConfig() const {
    PipelineConfig cfg;
    cfg.useGrayscale = grayscaleCheck_->isChecked();
    cfg.useBlur = blurCheck_->isChecked();
    cfg.blurKernel = sanitizeOddKernel(blurKernelSlider_->value());
    cfg.useCanny = cannyCheck_->isChecked();
    cfg.cannyLow = cannyLowSlider_->value();
    cfg.cannyHigh = cannyHighSlider_->value();
    cfg.useMotionDetect = motionCheck_->isChecked();
    cfg.minContourArea = minAreaSpin_->value();
    cfg.drawBoxes = boxesCheck_->isChecked();
    return cfg;
}

void MainWindow::onControlsChanged() {
    const PipelineConfig cfg = currentConfig();
    blurKernelValue_->setText(QString::number(cfg.blurKernel));
    cannyValue_->setText(QStringLiteral("%1 / %2")
                             .arg(static_cast<int>(cfg.cannyLow))
                             .arg(static_cast<int>(cfg.cannyHigh)));

    emit requestConfig(cfg);
    emit requestMaxFps(maxFpsSpin_->value());
}

void MainWindow::onStartCamera() {
    emit requestCamera(cameraIndexSpin_->value());
}

void MainWindow::onOpenFile() {
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("Open video"), QString(),
        QStringLiteral("Video files (*.mp4 *.avi *.mkv *.mov);;All files (*)"));
    if (path.isEmpty()) {
        return;
    }
    emit requestFile(path);
}

void MainWindow::onStop() {
    emit requestStop();
}

void MainWindow::openVideo(const QString& path) {
    emit requestFile(path);
}

void MainWindow::onFrameReady(const QImage& original, const QImage& processed,
                              FrameStats stats, double fps) {
    lastOriginal_ = original;
    lastProcessed_ = processed;
    updateImageLabels();

    fpsStatus_->setText(QStringLiteral("FPS: %1").arg(fps, 0, 'f', 1));
    timingStatus_->setText(QStringLiteral("frame: %1 ms  (pre %2 / det %3)")
                               .arg(stats.totalMs, 0, 'f', 2)
                               .arg(stats.preprocessMs, 0, 'f', 2)
                               .arg(stats.detectMs, 0, 'f', 2));
    detectionStatus_->setText(QStringLiteral("detections: %1").arg(stats.detections));
}

void MainWindow::onSourceOpened(const QString& description, int width, int height,
                                double sourceFps) {
    sourceStatus_->setText(QStringLiteral("source: %1  %2x%3 @ %4 fps")
                               .arg(description)
                               .arg(width)
                               .arg(height)
                               .arg(sourceFps, 0, 'f', 1));
    stopButton_->setEnabled(true);
    startCameraButton_->setEnabled(false);
}

void MainWindow::onSourceClosed() {
    sourceStatus_->setText(QStringLiteral("source: none"));
    stopButton_->setEnabled(false);
    startCameraButton_->setEnabled(true);
}

void MainWindow::onError(const QString& message) {
    QMessageBox::warning(this, QStringLiteral("Capture error"), message);
    onSourceClosed();
}

void MainWindow::updateImageLabels() {
    const auto apply = [](QLabel* label, const QImage& image) {
        if (image.isNull()) {
            return;
        }
        label->setPixmap(QPixmap::fromImage(image).scaled(
            label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    };
    apply(originalLabel_, lastOriginal_);
    apply(processedLabel_, lastProcessed_);
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    // Pencere buyudugunde son kareyi yeni boyuta gore olceklendir; yoksa
    // yeni kare gelene kadar kucuk goruntu ortada asili kalir.
    updateImageLabels();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    emit requestStop();
    QMainWindow::closeEvent(event);
}
