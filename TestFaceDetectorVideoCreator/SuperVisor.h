#ifndef SUPERVISOR_H
#define SUPERVISOR_H

#include <chrono>
#include <thread>
#include <map>

#include <QString>
#include <QObject>
#include <QImage>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <OpenCVStreamer.h>
#include <CpuFaceDetector.h>
#include <StreamDisplay.h>
#include <FaceMetaDataProcessor.h>

class SuperVisor : public QObject
{
    Q_OBJECT

public:
    SuperVisor(QString WorkDir);
    ~SuperVisor();

signals:
    void setVideoFrameSignal(const QImage&, const QSize&);
    void setFaceRectsSignal(const QVector<QRect>&);

    void getFaceRectsSignal(QVector<QRect>&);
    void getFrameNumSignal(int&);
    void getStreamDisplaySizeSignal(QSize&);
    void getPixmapSizeSignal(QSize&);

    void refreshApplicationSignal();

public slots:
    void setOpenCVStreamerFilePath(QString&, int&, bool&);

    void getNextVideoFrame(bool&);
    void getPrevVideoFrame(bool&);
    void getNumVideoFrame(const int&);

    void getDetectFaces();
    void runMaxSpeedMode();
    void stopMaxSpeedMode();

    void addVideoFacesMetaData();
    void saveMetaData(const QString&, bool&);
    void loadMetaData(const QString&, bool&);

    void clearState();
    void setFaceDetectorSize(const int&);

private:
//    void cvMatToQImage(const cv::Mat &img, QImage& qImg);

    OpenCVStreamer *_videoStream;
    CpuFaceDetector *_faceDetector;
    FaceMetaDataProcessor _faceMetaDataProcessor;

    cv::Mat _videoFrame;
    std::map<int, QVector<QRect>> _videoFacesMetaData;

    bool _isSpeedModeOn;
    bool *_isFrameProcessed;
};

#endif // SUPERVISOR_H
