#ifndef IMG_PROC_H
#define IMG_PROC_H

#include "boost_program_options.h"

#include <iostream>
#include <stdlib.h>
#include <string>

#include <dlib/array2d.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QDateTime>

using namespace std;

class img_proc
{
public:
    img_proc(); // конструктор

    // функция отображения изображения с разметкой на экране (в отдельном окне)
    static void display_image(dlib::matrix<uchar> imgGray, dlib::matrix<float> train);
    static void display_image(dlib::matrix<uchar> imgGray, dlib::matrix<float> train, dlib::matrix<float> test);

    // функции для закрытия глаз

    // афинное преобразование треугольной области
    static void triangle_affine(std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri, cv::Mat &img, cv::Mat &img2);
    // получить наборы координат закрытых/полузакрытых/прикрытих глаз
    static void get_eye_points(std::vector<cv::Point2f> &close, std::vector<cv::Point2f> &close2, dlib::matrix<float> &train, double coeff);
    // изменить глаз и набор его точек
    static void set_triangles_and_transform(cv::Mat &img, cv::Mat &img2, std::vector<cv::Point2f> &close, std::vector<cv::Point2f> &close2, int left_right);

    // функции изменения изображения

    // функция наложения изображения на фон (изображение смещается на случайное значение)
    static void put_on_background(cv::Mat &img, dlib::matrix<float> &train);
    // функция изменения размера изображения (с соблюдением пропорций сторон)
    static void resize_image(cv::Mat &img, dlib::matrix<float> &train);
    // функция отражения изображения по горизонтали
    static void flip_image(cv::Mat &img, dlib::matrix<float> &train);
    // функция для закрытия глаз (прикрыт, полузакрыт, закрыт)
    static void reshape_eyes(cv::Mat &img, dlib::matrix<float> &train);
    // функция для поворота изображения
    static void rotate(cv::Mat &img, dlib::matrix<float> &train);

    // функция для выбора патча левого глаза
    static void get_left_eye_patch(cv::Mat &img, cv::Mat &leye_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord);
    // функция для выбора патча правого глаза
    static void get_right_eye_patch(cv::Mat &img, cv::Mat &reye_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord);
    // функция для выбора патча носа
    static void get_nose_patch(cv::Mat &img, cv::Mat &nose_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord);
    // функция для выбора патча левой брови
    static void get_left_eyebow_patch(cv::Mat &img, cv::Mat &leyebow_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord);
    // функция для выбора патча правой брови
    static void get_right_eyebow_patch(cv::Mat &img, cv::Mat &reyebow_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord);
    // функция применения всех изменений
    static int transform_image(cv::Mat &img, dlib::matrix<float> &train);
    // функция проверки попадания точек в границы изображения (из-за поворота изображения некоторые точки могут выйти за границы)
    static bool borders_check(cv::Mat &img, dlib::matrix<float> &train);

};

#endif // IMG_PROC_H
