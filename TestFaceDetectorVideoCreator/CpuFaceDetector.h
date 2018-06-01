#ifndef CPUFACEDETECTOR_H
#define CPUFACEDETECTOR_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

static const int widenCoef = 3;
static const int narrowCoef = 8;
//constexpr

class CpuFaceDetector
{
public:
	struct FaceInfo
	{
		cv::Rect rect;
		cv::Rect widenRect;
                cv::Rect narrowRect;
	};

public:

	CpuFaceDetector(const std::string &nameFaceClassificator, int minSize = 50);

	~CpuFaceDetector(void);

    std::vector<CpuFaceDetector::FaceInfo> detect(const cv::Mat img) const;

	void setMinFaceSize(int size);

private:

	cv::Rect widenFaceRect(const cv::Rect& rc, const cv::Rect& frameRect) const;
        cv::Rect narrowFaceRect(const cv::Rect& rc, const cv::Rect& frameRect) const;

private:
	mutable cv::CascadeClassifier faceCascade;

	int faceMinSize;
};

#endif
