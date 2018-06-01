#include "FaceMetaDataProcessor.h"

FaceMetaDataProcessor::FaceMetaDataProcessor()
{
}

FaceMetaDataProcessor::~FaceMetaDataProcessor()
{
}

bool FaceMetaDataProcessor::serializeMetaData(const std::string &filePath, const std::map<int, QVector<QRect>> &faceMetaData)
{
    std::ofstream metaDataFile;

    metaDataFile.open(filePath, std::ios::out | std::ios::binary);

    if(!metaDataFile.is_open())
        return false;

    char header[] = "metadatafile";
    metaDataFile.write(header, sizeof(header));

    for(auto iter = faceMetaData.begin(); iter != faceMetaData.end(); ++iter)
    {
        int frameNum = iter->first;
        metaDataFile.write((char*)&frameNum, sizeof(frameNum));

        int faceRectsCount = iter->second.size();
        metaDataFile.write((char*)&faceRectsCount, sizeof(faceRectsCount));

        for(int i = 0; i < faceRectsCount; ++i)
        {
            int x = iter->second[i].x();
            int y = iter->second[i].y();
            int width = iter->second[i].width();
            int height = iter->second[i].height();

            metaDataFile.write((char*)&x, sizeof(x));
            metaDataFile.write((char*)&y, sizeof(y));
            metaDataFile.write((char*)&width, sizeof(width));
            metaDataFile.write((char*)&height, sizeof(height));
        }
    }

    int finalSimbol = -1;
    metaDataFile.write((char*)&finalSimbol, sizeof(finalSimbol));

    metaDataFile.close();

    return true;
}

bool FaceMetaDataProcessor::deserializeMetaData(const std::string &filePath, std::map<int, QVector<QRect>> &faceMetaData)
{
    faceMetaData.clear();

    std::ifstream metaDataFile;

    metaDataFile.open(filePath, std::ios::in | std::ios::binary);

    if(!metaDataFile.is_open())
        return false;

    char header[13];
    metaDataFile.read(header, sizeof(header));

    if(strcmp(header, "metadatafile") != 0)
        return false;

    int frameNum;
    metaDataFile.read((char*)&frameNum, sizeof(frameNum));

    while(frameNum != -1)
    {
        QVector<QRect> frameFaceRects;
        faceMetaData[frameNum] = frameFaceRects;

        int faceRectsCount;
        metaDataFile.read((char*)&faceRectsCount, sizeof(faceRectsCount));

        for(int i = 0; i < faceRectsCount; ++i)
        {
            int x;
            int y;
            int width;
            int height;

            metaDataFile.read((char*)&x, sizeof(x));
            metaDataFile.read((char*)&y, sizeof(y));
            metaDataFile.read((char*)&width, sizeof(width));
            metaDataFile.read((char*)&height, sizeof(height));

            QRect faceRect(x, y, width, height);

            frameFaceRects.push_back(faceRect);
        }

        faceMetaData[frameNum] = frameFaceRects;

        metaDataFile.read((char*)&frameNum, sizeof(frameNum));
    }

    metaDataFile.close();

    return true;
}

/*void FaceMetaDataProcessor::deserializeMetaDataToStdOpenCV(const std::string &filePath, std::map<int, std::vector<cv::Rect>> &faceMetaData)
{

}*/
