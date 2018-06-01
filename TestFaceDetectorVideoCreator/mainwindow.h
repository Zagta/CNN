#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QPainter>
#include <QMessageBox>
#include <QCloseEvent>

#include <SuperVisor.h>
#include <StreamDisplay.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QString WorkDir, SuperVisor *superVisor, QWidget *parent = 0);
    ~MainWindow();

signals:
    void setOpenCVStreamerFilePathSignal(QString&, int&,  bool&);
    void getNextVideoFrameSignal(bool&);
    void getPrevVideoFrameSignal(bool&);
    void getNumVideoFrameSignal(const int&);

    void getDetectFacesSignal();
    void addVideoFacesMetaDataSignal();
    void runMaxSpeedModeSignal();
    void stopMaxSpeedModeSignal();

    void saveMetaDataSignal(const QString&, bool&);
    void loadMetaDataSignal(const QString&, bool&);

    void clearStateSignal();
    void setFaceDetectorSizeSignal(const int&);

private slots:
    void on_openVideoFileButton_clicked();

    void on_nextFrameButton_clicked();

    void on_saveMetaDataButton_clicked();

    void on_loadMetaDataButton_clicked();

    void on_prevFrameButton_clicked();

    void on_speedModeButton_clicked();

    void refreshApplication();

    void on_stopSpeedModeButton_clicked();

    void on_videoSlider_sliderReleased();

    void on_videoPositionLineEdit_editingFinished();

    void on_faceDetectorSizeLineEdit_editingFinished();

private:
    Ui::MainWindow *ui;

    StreamDisplay *_streamDisplay;
    QString sWorkDir;

    bool _isStreamOpen;

    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
