#include "CpuFaceDetector.h"

CpuFaceDetector::CpuFaceDetector(const std::string &nameFaceClassificator, int minSize) : faceMinSize(minSize)
{
	if (!faceCascade.load(nameFaceClassificator))
        std::cout << "Not loaded classificator" << std::endl;
	else
        std::cout << "Classificator loaded" << std::endl;
}

CpuFaceDetector::~CpuFaceDetector(void)
{	
}

std::vector<CpuFaceDetector::FaceInfo> CpuFaceDetector::detect(const cv::Mat img) const
{
    std::vector<FaceInfo> faces;

  	cv::Mat resImg;
  	cv::Size size;
 
	size.height = img.rows;
  	size.width = img.cols;

	const cv::Rect frameRect(0, 0, img.cols, img.rows);

	cv::Mat frame_gray;

	if(img.channels() == 3)
		cvtColor(img, frame_gray, cv::COLOR_BGR2GRAY);
	else
		frame_gray = img;

    cv::equalizeHist(frame_gray, resImg);

	std::vector<cv::Rect> foundedFaces;

    faceCascade.detectMultiScale(resImg, foundedFaces, 1.2, 2, 0 | CV_HAAR_FEATURE_MAX, cv::Size(faceMinSize, faceMinSize));

	for (size_t i = 0; i < foundedFaces.size(); ++i)
	{
		FaceInfo info;
		info.rect = foundedFaces[i];
		info.widenRect = widenFaceRect(foundedFaces[i], frameRect);
        info.narrowRect = narrowFaceRect(foundedFaces[i], frameRect);

        faces.push_back(info);
    }

    return faces;
}

cv::Rect CpuFaceDetector::widenFaceRect(const cv::Rect& rc, const cv::Rect& frameRect) const
{
	cv::Rect res = rc;

    int dx = res.width / widenCoef;
    int dy = res.height / widenCoef;

	res.x -= dx;
	res.y -= dy;

	res.width += 2 * dx;
	res.height += 2 * dy;

	if( res.x < 0 )
		res.x = 0;

	if( res.y < 0 )
		res.y = 0;

	if(res.x + res.width > frameRect.width)
		res.width = frameRect.width - res.x;

	if(res.y + res.height > frameRect.height)
		res.height = frameRect.height - res.y;

	return res;
}

cv::Rect CpuFaceDetector::narrowFaceRect(const cv::Rect& rc, const cv::Rect& frameRect) const
{
    cv::Rect res = rc;

    int dx = res.width / narrowCoef;
    int dy = res.height / narrowCoef;

    //res.x += dx;
    res.y += dy;

    //res.width -= 2 * dx;
    //res.height -= 2 * dy;

    if(res.x + res.width > frameRect.width)
        res.width = frameRect.width - res.x;

    if(res.y + res.height > frameRect.height)
        res.height = frameRect.height - res.y;

    return res;
}

void CpuFaceDetector::setMinFaceSize( int size )
{
	faceMinSize = size;
}
