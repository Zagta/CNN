#include "OpenCVStreamer.h"

OpenCVStreamer::OpenCVStreamer()
{
    _frameNum = -1;
    _frameCount = 0;
}

OpenCVStreamer::~OpenCVStreamer()
{
}

void OpenCVStreamer::setFilePath(std::string &path)
{
    _filePath = path;
}

bool OpenCVStreamer::openVideoStream()
{
    return openVideoStream(_filePath);
}

bool OpenCVStreamer::openVideoStream(std::string &path)
{
    _frameNum = -1;
    _frameCount = 0;

    bool isVideoStreamOpen;
    isVideoStreamOpen = _videoStream.open(path);
    _videoStream.set(CV_CAP_PROP_FPS, 100);

    if(isVideoStreamOpen)
    {
        _frameCount = _videoStream.get(CV_CAP_PROP_FRAME_COUNT);
        return true;
    }
    else
    {
        return false;
    }
}

bool OpenCVStreamer::getNextVideoFrame(cv::Mat &img)
{
    if(_frameNum == _frameCount)
        return false;

    ++_frameNum;

    return _videoStream.read(img);
}

bool OpenCVStreamer::getPrevVideoFrame(cv::Mat &img)
{
    if(_frameNum == 0)
        return false;

    --_frameNum;

    _videoStream.set(CV_CAP_PROP_POS_FRAMES, _frameNum);
    return _videoStream.read(img);
}

bool OpenCVStreamer::getNumVideoFrame(const int &frameNum, cv::Mat &img)
{
    if(frameNum < 0 || frameNum > _frameCount)
        return false;

    _frameNum = frameNum;

    _videoStream.set(CV_CAP_PROP_POS_FRAMES, _frameNum);
    return _videoStream.read(img);
}

void OpenCVStreamer::getFrameNum(int &frameNum)
{
    frameNum = _frameNum;
}

void OpenCVStreamer::getFrameCount(int &frameCount)
{
    frameCount = _frameCount;
}

void OpenCVStreamer::closeStream()
{
    _videoStream.release();
}
