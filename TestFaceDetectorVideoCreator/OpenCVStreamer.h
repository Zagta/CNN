#ifndef OPENCVSTREAMER_H
#define OPENCVSTREAMER_H

#include <QObject>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Обёртка над VideoCapture
class OpenCVStreamer
{
public:
    OpenCVStreamer();
    ~OpenCVStreamer();

    void setFilePath(std::string &path);

    bool openVideoStream();
    bool openVideoStream(std::string &path);

    bool getNextVideoFrame(cv::Mat &img);
    bool getPrevVideoFrame(cv::Mat &img);
    bool getNumVideoFrame(const int &frameNum, cv::Mat &img);

    void getFrameNum(int&);
    void getFrameCount(int&);

    void closeStream();

private:
    std::string _filePath;

    int _frameNum;
    int _frameCount;

    cv::VideoCapture _videoStream;
};

#endif // OPENCVSTREAMER_H
