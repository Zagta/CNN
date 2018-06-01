#include "SuperVisor.h"

SuperVisor::SuperVisor(QString WorkDir)
{
    _faceDetector = new CpuFaceDetector(WorkDir.toStdString()+"/haarcascade_frontalface_alt.xml", 100);
    _videoStream = new OpenCVStreamer();
}

SuperVisor::~SuperVisor()
{
    delete[] _isFrameProcessed;
    delete _faceDetector;
    delete _videoStream;
}

void SuperVisor::setOpenCVStreamerFilePath(QString &filePath, int &frameCount, bool &isStreamOpen)
{
    // открываем видео файл
    std::string filePathStr = filePath.toLocal8Bit().constData();
    _videoStream->setFilePath(filePathStr);
    isStreamOpen = _videoStream->openVideoStream();
    if(isStreamOpen)
    {
        _videoStream->getFrameCount(frameCount);
        _isFrameProcessed = new bool[frameCount];
        for(int i = 0; i < frameCount; ++i)
            _isFrameProcessed[i] = false;
    }
    else
    {
        frameCount = 0;
    }
}

void SuperVisor::getNextVideoFrame(bool &isFrameGet)
{
    QSize streamDisplaySize;
    emit getStreamDisplaySizeSignal(streamDisplaySize);

    if(!_videoStream->getNextVideoFrame(_videoFrame))
    {
        isFrameGet = false;
        return;
    }

    cv::cvtColor(_videoFrame, _videoFrame, CV_BGR2RGB);
    QImage qImg((uchar*)_videoFrame.data, _videoFrame.cols, _videoFrame.rows, QImage::Format_RGB888);

    QImage qImgScaled = qImg.scaled(streamDisplaySize.width(), streamDisplaySize.height(), Qt::KeepAspectRatio);

    QSize videoFrameSize(_videoFrame.cols, _videoFrame.rows);

    emit setVideoFrameSignal(qImgScaled, videoFrameSize);

    getDetectFaces();

    isFrameGet = true;
}

void SuperVisor::getPrevVideoFrame(bool &isFrameGet)
{
    QSize streamDisplaySize;
    emit getStreamDisplaySizeSignal(streamDisplaySize);

    if(!_videoStream->getPrevVideoFrame(_videoFrame))
    {
        isFrameGet = false;
        return;
    }

    cv::cvtColor(_videoFrame, _videoFrame, CV_BGR2RGB);
    QImage qImg((uchar*)_videoFrame.data, _videoFrame.cols, _videoFrame.rows, QImage::Format_RGB888);

    QImage qImgScaled = qImg.scaled(streamDisplaySize.width(), streamDisplaySize.height(), Qt::KeepAspectRatio);

    QSize videoFrameSize(_videoFrame.cols, _videoFrame.rows);

    emit setVideoFrameSignal(qImgScaled, videoFrameSize);

    getDetectFaces();

    isFrameGet = true;
}

void SuperVisor::getNumVideoFrame(const int &frameNum)
{
    QSize streamDisplaySize;
    emit getStreamDisplaySizeSignal(streamDisplaySize);

    if(!_videoStream->getNumVideoFrame(frameNum, _videoFrame))
        return;

    cv::cvtColor(_videoFrame, _videoFrame, CV_BGR2RGB);
    QImage qImg((uchar*)_videoFrame.data, _videoFrame.cols, _videoFrame.rows, QImage::Format_RGB888);

    QImage qImgScaled = qImg.scaled(streamDisplaySize.width(), streamDisplaySize.height(), Qt::KeepAspectRatio);

    QSize videoFrameSize(_videoFrame.cols, _videoFrame.rows);

    emit setVideoFrameSignal(qImgScaled, videoFrameSize);

    getDetectFaces();
}

void SuperVisor::getDetectFaces()
{
    QSize pixmapSize;
    emit getPixmapSizeSignal(pixmapSize);

    float widthRatio = float(_videoFrame.cols) / float(pixmapSize.width());
    float heightRatio = float(_videoFrame.rows) / float(pixmapSize.height());

    int frameNum;
    _videoStream->getFrameNum(frameNum);

    QVector<QRect> qFaceRects;

    // если имеется информация о найденых лицах, выводим прямоугольники
    if(_videoFacesMetaData.find(frameNum) != _videoFacesMetaData.end())
    {
        qFaceRects = _videoFacesMetaData[frameNum];

        QVector<QRect> qFaceRectsScaled;
        for(int i = 0; i < qFaceRects.size(); ++i)
        {
            QRect qFaceRect(int(qFaceRects[i].x() / widthRatio + 0.5),
                            int(qFaceRects[i].y() / heightRatio + 0.5),
                            int(qFaceRects[i].width() / widthRatio + 0.5),
                            int(qFaceRects[i].height() / heightRatio + 0.5));

            qFaceRectsScaled.push_back(qFaceRect);
        }

        emit setFaceRectsSignal(qFaceRectsScaled);
    }
    // если нет данных о лицах на кадре, пытаем найти лица с помощью детектора
    else
    {
        if(!_isFrameProcessed[frameNum])
        {
            std::vector<CpuFaceDetector::FaceInfo> faceInfo = _faceDetector->detect(_videoFrame);

            for(int i = 0; i < faceInfo.size(); ++i)
            {
                cv::Rect faceRect = faceInfo[i].rect;
                QRect qFaceRect(faceRect.x / widthRatio, faceRect.y / heightRatio, faceRect.width / widthRatio, faceRect.height / heightRatio);

                qFaceRects.push_back(qFaceRect);
            }
        }

        emit setFaceRectsSignal(qFaceRects);
    }
}

void SuperVisor::addVideoFacesMetaData()
{
    QSize streamDisplaySize;
    emit getStreamDisplaySizeSignal(streamDisplaySize);

    float widthRatio = float(_videoFrame.cols) / float(streamDisplaySize.width());
    float heightRatio = float(_videoFrame.rows) / float(streamDisplaySize.height());

    QVector<QRect> qFaceRects;
    emit getFaceRectsSignal(qFaceRects);

    int frameNum;
    _videoStream->getFrameNum(frameNum);

    // сохраняем метаданные для найденных прямоугольников
    if(qFaceRects.size() != 0)
    {
        QVector<QRect> qFaceRectsRescaled;
        for(int i = 0; i < qFaceRects.size(); ++i)
        {
            QRect qFaceRect(int(qFaceRects[i].x() * widthRatio + 0.5),
                            int(qFaceRects[i].y() * heightRatio + 0.5),
                            int(qFaceRects[i].width() * widthRatio + 0.5),
                            int(qFaceRects[i].height() * heightRatio + 0.5));

            qFaceRectsRescaled.push_back(qFaceRect);
        }
        _videoFacesMetaData[frameNum] = qFaceRectsRescaled;
    }
    // если прямоугольники были удалены, стираем информацию о них
    else
    {
        if(_videoFacesMetaData.find(frameNum) != _videoFacesMetaData.end())
            _videoFacesMetaData.erase(frameNum);
    }

    _isFrameProcessed[frameNum] = true;
}

void SuperVisor::saveMetaData(const QString &filePath, bool &isMetaDataFileSave)
{
    isMetaDataFileSave = _faceMetaDataProcessor.serializeMetaData(filePath.toLocal8Bit().constData(), _videoFacesMetaData);
}

void SuperVisor::loadMetaData(const QString &filePath, bool &isMetaDataFileOpen)
{
    isMetaDataFileOpen = _faceMetaDataProcessor.deserializeMetaData(filePath.toLocal8Bit().constData(), _videoFacesMetaData);

    if(isMetaDataFileOpen)
        getDetectFaces();
}

void SuperVisor::runMaxSpeedMode()
{
    _isSpeedModeOn = true;

    while(true)
    {
        if(!_isSpeedModeOn)
            break;

        addVideoFacesMetaData();

        bool isFrameGet;
        getNextVideoFrame(isFrameGet);
        if(!isFrameGet)
            break;

        emit refreshApplicationSignal();
    }
}

void SuperVisor::stopMaxSpeedMode()
{
    _isSpeedModeOn = false;
}

void SuperVisor::clearState()
{
    delete[] _isFrameProcessed;
    _videoFacesMetaData.clear();
    _videoStream->closeStream();
}

void SuperVisor::setFaceDetectorSize(const int &faceSize)
{
    _faceDetector->setMinFaceSize(faceSize);
}
