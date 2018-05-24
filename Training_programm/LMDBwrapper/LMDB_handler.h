#ifndef LMDB_HANDLER_H
#define LMDB_HANDLER_H

#include "LMDB.h"

#include "boost_program_options.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <bitset>

#include <QDir>
#include <QtCore>
#include <QStringList>
#include <QTime>


#include <dlib/serialize.h>
#include <dlib/data_io.h>

class LMDB_handler
{
public:
    LMDB_handler();
    static LMDB* NewLMDB();

    // сериализия изображения с вектором точек в строку (изображение матрица cv, вектор векторов из шортов [координаты точек в шорте], преобразованая строка с вектором и изображением)
    static void serialize8UMatWithVecLabelToStr(const cv::Mat& img, const std::vector<std::vector<short>> &labelVector, std::string& matWithVectorAsStr);

    // десериализация полученных строк обратно в изображения
    static void deserializeStrTo8UMatAndVecLabel(const std::string& matWithVectorAsStr, cv::Mat& img,  std::vector<std::vector<short>>  &labelVector);

    // заполнение базы даннных
    static void fillLMDB();
};

#endif // LMDB_HANDLER_H
