#ifndef BOOST_PROGRAM_OPTIONS_H
#define BOOST_PROGRAM_OPTIONS_H

#include <iostream>

#include "boost/scoped_ptr.hpp"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time.hpp>

#include <dlib/array2d.h>

using namespace std;

namespace po = boost::program_options;

class boost_program_options
{
public:
    boost_program_options();

    // параметры командной строки

    static std::string data_folder_path;                                       // путь к данных
    static std::string lmdb_folder_path;                                       // путь к базе
    static std::string snapshot_directory_path;                                // директория для снимков
    static std::string background_directory_path;                              // директория для фонов
    static std::string network_name;                                           // имя сети
    static std::string iter_name;                                              // номер итерации для снапшота
    static int mini_batch_size;                                                // размер мини-пакета
    static float learning_rate;                                                // коэффициент обучения
    static float required_learning_rate;                                       // требуемый коэффициент обучения
    static float learning_rate_shrink;                                         // шаг уменьшения коэффициента обучения
    static int iteration_without_progress;                                     // число итераций без прогресса
    static float weight_decay;                                                 // уменьшение веса
    static bool backgrounds;                                                   // использование различных фонов
    static bool random_cropper_enabled;                                        // использование изменения изображения и повороты
    static bool random_cropper_resize;                                         // использование изменения размера изображения
    static bool random_cropper_flip;                                           // использование отражения изображения по горизонтали
    static bool eye_reshaper;                                                  // использование eye_reshaper
    static int eye_chance;                                                     // вероятность закрытого глаза
    static bool multiply_network;                                              // использование второй сети
    static bool check_image;                                                   // проверка изображения
    static int images;                                                         // 42100; Helen - 1605; celba4500 - 3600; celeba+helens - 4600; lfwp - 600; accurate300wDetect - 2000; accurate300WCalc - 12000; accurate300wPlusCalc - 18500; celebafull - 530000 (всего 547221);
    static int multiply_network_number;                                        // номер используемой сети
    static int cascade_num;                                                    // параметр для обучения каскада
    static std::string draft_network_name;                                     // имя черновой сети
    static std::string draft_iter_name;                                        // номер итерации для чернового снапшота

    static dlib::matrix<int,2,5> patches_sizes;                                // размеры патчей для частей исходного изображения (брови, нос, глаза)
    static dlib::matrix<int,1,5> points_sizes;                                 // количество точек в патчах исходного изображения (брови, нос, глаза)

    // функция для парсинга параметров из командной строки
    static void SetParams(po::variables_map &vm);
    static int GetParams(int argc, char *argv[]);


};

#endif // BOOST_PROGRAM_OPTIONS_H
