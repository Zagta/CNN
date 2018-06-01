#ifndef FACEMETADATAPROCESSOR_H
#define FACEMETADATAPROCESSOR_H

#include <map>
#include <fstream>

#include <QVector>
#include <QRect>

#include <opencv2/core/core.hpp>

// класс обработчик файлов метаданных
class FaceMetaDataProcessor
{
public:
    FaceMetaDataProcessor();
    ~FaceMetaDataProcessor();

    bool serializeMetaData(const std::string &filePath, const std::map<int, QVector<QRect>> &faceMetaData);

    bool deserializeMetaData(const std::string &filePath, std::map<int, QVector<QRect>> &faceMetaData);
    //void deserializeMetaDataToStdOpenCV(const std::string &filePath, std::map<int, std::vector<cv::Rect>> &faceMetaData); //TODO
};

#endif // FACEMETADATAPROCESSOR_H
