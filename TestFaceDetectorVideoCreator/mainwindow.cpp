#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QString>

MainWindow::MainWindow(QString WorkDir, SuperVisor *superVisor, QWidget *parent) :
    QMainWindow(parent),
    sWorkDir(WorkDir),
    ui(new Ui::MainWindow)
{
    _isStreamOpen = false;

    _streamDisplay = new StreamDisplay();
    _streamDisplay->setGeometry(QRect(10, 10, 1021, 571));
    _streamDisplay->setMinimumSize(100, 100);
    _streamDisplay->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    connect(this, SIGNAL(setOpenCVStreamerFilePathSignal(QString&, int&, bool&)), superVisor, SLOT(setOpenCVStreamerFilePath(QString&, int&, bool&)));
    connect(this, SIGNAL(getNextVideoFrameSignal(bool&)), superVisor, SLOT(getNextVideoFrame(bool&)));
    connect(this, SIGNAL(getPrevVideoFrameSignal(bool&)), superVisor, SLOT(getPrevVideoFrame(bool&)));
    connect(this, SIGNAL(getNumVideoFrameSignal(const int&)), superVisor, SLOT(getNumVideoFrame(const int&)));

    connect(this, SIGNAL(addVideoFacesMetaDataSignal()), superVisor, SLOT(addVideoFacesMetaData()));
    connect(this, SIGNAL(saveMetaDataSignal(const QString&, bool&)), superVisor, SLOT(saveMetaData(const QString&, bool&)));
    connect(this, SIGNAL(loadMetaDataSignal(const QString&, bool&)), superVisor, SLOT(loadMetaData(const QString&, bool&)));
    connect(this, SIGNAL(runMaxSpeedModeSignal()), superVisor, SLOT(runMaxSpeedMode()));
    connect(this, SIGNAL(stopMaxSpeedModeSignal()), superVisor, SLOT(stopMaxSpeedMode()));
    connect(this, SIGNAL(clearStateSignal()), superVisor, SLOT(clearState()));
    connect(this, SIGNAL(setFaceDetectorSizeSignal(const int&)), superVisor, SLOT(setFaceDetectorSize(const int&)));

    connect(superVisor, SIGNAL(refreshApplicationSignal()), this, SLOT(refreshApplication()));

    connect(superVisor, SIGNAL(setVideoFrameSignal(const QImage&, const QSize&)), _streamDisplay, SLOT(setVideoFrame(const QImage&, const QSize&)));
    connect(superVisor, SIGNAL(setFaceRectsSignal(const QVector<QRect>&)), _streamDisplay, SLOT(setFaceRects(const QVector<QRect>&)));
    connect(superVisor, SIGNAL(getFaceRectsSignal(QVector<QRect>&)), _streamDisplay, SLOT(getFaceRects(QVector<QRect>&)));
    connect(superVisor, SIGNAL(getStreamDisplaySizeSignal(QSize&)), _streamDisplay, SLOT(getStreamDisplaySize(QSize&)));
    connect(superVisor, SIGNAL(getPixmapSizeSignal(QSize&)), _streamDisplay, SLOT(getPixmapSize(QSize&)));

    ui->setupUi(this);

    ui->streamDisplayLayout->addWidget(_streamDisplay);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_openVideoFileButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open video stream"), sWorkDir, tr("All files (*)"));

    if(_isStreamOpen)
        emit clearStateSignal();

    int frameCount;
    emit setOpenCVStreamerFilePathSignal(fileName, frameCount, _isStreamOpen);

    if(_isStreamOpen)
    {
        ui->faceDetectorSizeLineEdit->setEnabled(true);
        ui->videoPositionLineEdit->setEnabled(true);
        ui->videoPositionLineEdit->setText("0");
        ui->videoSlider->setEnabled(true);
        ui->videoSlider->setMaximum(frameCount);
        ui->videoSlider->setValue(0);

        ui->speedModeButton->setEnabled(true);
        ui->nextFrameButton->setEnabled(true);
        ui->prevFrameButton->setEnabled(true);
        ui->saveMetaDataButton->setEnabled(true);
        ui->loadMetaDataButton->setEnabled(true);

        bool isFrameGet;
        emit getNextVideoFrameSignal(isFrameGet);
    }
    else
    {
        QMessageBox streamNotOpenMessage;
        streamNotOpenMessage.setIcon(QMessageBox::Warning);
        streamNotOpenMessage.setText("Проблема при открытии видео файла");
        streamNotOpenMessage.setInformativeText("Не удалось открыть видео файл:\n" + fileName + ".\nПроверьте корректность выбранного пути.");
        streamNotOpenMessage.exec();
    }
}

void MainWindow::on_nextFrameButton_clicked()
{
    emit addVideoFacesMetaDataSignal();

    bool isFrameGet;
    emit getNextVideoFrameSignal(isFrameGet);
    if(isFrameGet)
    {
        int videoSliderValue = ui->videoSlider->value();
        ui->videoSlider->setValue(videoSliderValue + 1);
        ui->videoPositionLineEdit->setText(QString(std::to_string(videoSliderValue + 1).c_str()));
    }
}

void MainWindow::on_prevFrameButton_clicked()
{
    emit addVideoFacesMetaDataSignal();

    bool isFrameGet;
    emit getPrevVideoFrameSignal(isFrameGet);
    if(isFrameGet)
    {
        int videoSliderValue = ui->videoSlider->value();
        ui->videoSlider->setValue(videoSliderValue - 1);
        ui->videoPositionLineEdit->setText(QString(std::to_string(videoSliderValue - 1).c_str()));
    }
}

void MainWindow::on_saveMetaDataButton_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save metadata file"), sWorkDir, tr("All files (*)"));

    bool isMetaDataFileSave;
    emit saveMetaDataSignal(fileName, isMetaDataFileSave);

    if(!isMetaDataFileSave)
    {
        QMessageBox metaDataFileNotSaveMessage;
        metaDataFileNotSaveMessage.setIcon(QMessageBox::Warning);
        metaDataFileNotSaveMessage.setText("Проблема при сохранении файла метаданных");
        metaDataFileNotSaveMessage.setInformativeText("Не удалось сохранить файл мета данных:\n" + fileName + ".");
        metaDataFileNotSaveMessage.exec();
    }
}

void MainWindow::on_loadMetaDataButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open metadata file"), sWorkDir, tr("All files (*)"));

    bool isMetaDataFileOpen;
    emit loadMetaDataSignal(fileName, isMetaDataFileOpen);

    if(!isMetaDataFileOpen)
    {
        QMessageBox metaDataFileNotOpenMessage;
        metaDataFileNotOpenMessage.setIcon(QMessageBox::Warning);
        metaDataFileNotOpenMessage.setText("Проблема при открытии файла метаданных");
        metaDataFileNotOpenMessage.setInformativeText("Не удалось открыть файл мета данных:\n" + fileName + ".\nПроверьте корректность выбранного пути.");
        metaDataFileNotOpenMessage.exec();
    }
}

void MainWindow::on_speedModeButton_clicked()
{
    ui->stopSpeedModeButton->setEnabled(true);

    ui->faceDetectorSizeLineEdit->setEnabled(false);
    ui->openVideoFileButton->setEnabled(false);
    ui->videoPositionLineEdit->setEnabled(false);
    ui->videoSlider->setEnabled(false);
    ui->speedModeButton->setEnabled(false);

    ui->loadMetaDataButton->setEnabled(false);
    ui->saveMetaDataButton->setEnabled(false);
    ui->nextFrameButton->setEnabled(false);
    ui->prevFrameButton->setEnabled(false);

    emit runMaxSpeedModeSignal();
}

void MainWindow::on_stopSpeedModeButton_clicked()
{
    ui->stopSpeedModeButton->setEnabled(false);

    ui->faceDetectorSizeLineEdit->setEnabled(true);
    ui->openVideoFileButton->setEnabled(true);
    ui->videoPositionLineEdit->setEnabled(true);
    ui->videoSlider->setEnabled(true);
    ui->speedModeButton->setEnabled(true);

    ui->loadMetaDataButton->setEnabled(true);
    ui->saveMetaDataButton->setEnabled(true);
    ui->nextFrameButton->setEnabled(true);
    ui->prevFrameButton->setEnabled(true);

    emit stopMaxSpeedModeSignal();
}

void MainWindow::refreshApplication()
{
    int videoSliderValue = ui->videoSlider->value();
    ui->videoSlider->setValue(videoSliderValue + 1);
    ui->videoPositionLineEdit->setText(QString(std::to_string(videoSliderValue + 1).c_str()));

    qApp->processEvents();
}


void MainWindow::on_videoSlider_sliderReleased()
{
    int videoPositionInt = ui->videoSlider->value();
    QString videoPositionQStr = QString(std::to_string(videoPositionInt).c_str());
    ui->videoPositionLineEdit->setText(videoPositionQStr);

    emit getNumVideoFrameSignal(videoPositionInt);
}

void MainWindow::on_videoPositionLineEdit_editingFinished()
{
    QString videoPositionQStr = ui->videoPositionLineEdit->text();
    int videoPositionInt = videoPositionQStr.toInt();

    if(videoPositionInt < 0)
    {
        videoPositionInt = 0;
        ui->videoPositionLineEdit->setText("0");
    }
    else if(videoPositionInt > ui->videoSlider->maximum())
    {
        videoPositionInt = ui->videoSlider->maximum();
        ui->videoPositionLineEdit->setText(QString(std::to_string(ui->videoSlider->maximum()).c_str()));
    }

    ui->videoSlider->setValue(videoPositionInt);

    emit getNumVideoFrameSignal(videoPositionInt);
}

void MainWindow::on_faceDetectorSizeLineEdit_editingFinished()
{
    int faceDetectorSize = ui->faceDetectorSizeLineEdit->text().toInt();

    emit setFaceDetectorSizeSignal(faceDetectorSize);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    emit stopMaxSpeedModeSignal();
}
