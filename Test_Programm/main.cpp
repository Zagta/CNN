#include <QCoreApplication>

#include "LMDB.h"
#include "CpuFaceDetector.h"

#include <lmdb.h>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <cmath>

#include <dlib/array2d.h>
#include <dlib/serialize.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/opencv/cv_image.h>

#include "boost/scoped_ptr.hpp"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time.hpp>

#include <bitset>
#include <math.h>

#include <QtCore>
#include <QDir>
#include <QStringList>
#include <QTime>

#include <QString>

using namespace std;
using namespace dlib;
using namespace cv;

std::string snapshot_directory_path;                                // директория для снимков
std::string network_name;                                           // имя сети
std::string iter_name;                                              // номер итерации для снапшота
std::string video_path;                                             // путь к видео или значение для вебкамеры
int net_num;
int multiply_network_number;                                        // номер используемой сети
std::string draft_network_name;                                     // имя черновой сети
std::string draft_iter_name;                                        // номер итерации для чернового снапшота
int cascade_num;                                                    // параметр для каскада
std::string network_name_leb;                                       // имя сети для левой брови
std::string iter_name_leb;                                          // номер итерации для сети левой брови
std::string network_name_reb;                                       // имя сети для правой брови
std::string iter_name_reb;                                          // номер итерации для сети правой брови
std::string network_name_nose;                                      // имя сети для носа
std::string iter_name_nose;                                         // номер итерации для сети носа
std::string network_name_le;                                        // имя сети для левого глаза
std::string iter_name_le;                                           // номер итерации для сети левого глаза
std::string network_name_re;                                        // имя сети для правого глаза
std::string iter_name_re;                                           // номер итерации для сети правого глаза
// размеры для патчей бровей, носа и глаз
dlib::matrix<int,2,5> patches_sizes;
// количество точек
dlib::matrix<int,1,5> points_sizes;
// параметр для видеофайла, папки с изображениями или номера вебкамеры для проверки
string webcam_parametre = "/home/ginseng/BaseForTest/FERET";
namespace po = boost::program_options;

// --- СЕТИ ДЛЯ ПРОВЕРКИ ---
// сеть 7
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<50,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<45,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<35,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<30,5,5,2,2,
                            max_pool<2,2,2,2,relu<bn_con<con<20,15,15,3,3,
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 8
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<160,2,2,1,1,  // (2x2 s1) 4x4-> 2x2 / (2x2 s1) 1x1 -> 1x1
                            max_pool<2,2,2,2,relu<bn_con<con<80,2,2,1,1,   // (2x2 s1) 10x10 -> 5x5 / (2x2 s1) 4x4 -> 2x2
                            max_pool<2,2,2,2,relu<bn_con<con<40,3,3,1,1,   // (3x3 s1) 21x21 -> 11x11 / (3x3 s1) 9x9 -> 5x5
                            max_pool<2,2,2,2,relu<bn_con<con<20,5,5,2,2,   // (5x5 s2) 45x45 -> 23x23 / (5x5 s2) 22x22 -> 11x11
                            max_pool<2,2,2,2,relu<bn_con<con<15,16,16,1,1, // (16x16 s1) 185x185 -> 93x93 / (16x16 s2) 93x93 -> 47x47
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 9
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<256,3,3,1,1,   // 9x9 -> 5x5
                            max_pool<2,2,2,2,relu<bn_con<con<128,3,3,1,1,   // 22x22 -> 11x11
                            max_pool<2,2,2,2,relu<bn_con<con<64,3,3,1,1,    // 47x47 -> 24x24
                            max_pool<2,2,2,2,relu<bn_con<con<32,3,3,1,1,    // 97x97 -> 49x49
                            max_pool<2,2,2,2,relu<bn_con<con<16,3,3,1,1,    // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 10
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<256,3,3,1,1,   // 9x9 -> 5x5
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,   // 22x22 -> 11x11
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,    // 47x47 -> 24x24
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,    // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,    // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 11
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,   // 9x9 -> 5x5
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,   // 22x22 -> 11x11
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,    // 47x47 -> 24x24
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,    // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<8,3,3,1,1,    // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;
// сеть 12
/*using net_type_2 = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,  // 9x9 -> 5x5
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,  // 22x22 -> 11x11
                            prelu<bn_con<con<64,3,3,1,1,
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,   // 47x47 -> 24x24
                            prelu<bn_con<con<16,3,3,1,1,
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,   // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<8,3,3,1,1,   // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 15
using net_type_3 = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 7x7 -> 4x4
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 17x17 -> 9x9
                            max_pool<2,2,2,2,prelu<bn_con<con<16,5,5,1,1,       // 37x37 -> 19x19
                            max_pool<2,2,2,2,prelu<bn_con<con<16,10,10,1,1,     // 82x82 -> 41x41
                            prelu<bn_con<con<8,20,20,2,2,                       // 91x91
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>;
// сеть 16
using net_type_2 = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 16x16 -> 8x8
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 35x35 -> 18x18
                            prelu<bn_con<con<16,5,5,1,1,                        // 37x37
                            max_pool<2,2,2,2,prelu<bn_con<con<16,10,10,1,1,     // 82x82 -> 41x41
                            prelu<bn_con<con<8,20,20,2,2,                       // 91x91
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>;

// --- СЕТИ ДЛЯ ПРОВЕРКИ КАСКАДА
// черновая сеть 15
using draft_net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 7x7 -> 4x4
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 17x17 -> 9x9
                            max_pool<2,2,2,2,prelu<bn_con<con<16,5,5,1,1,       // 37x37 -> 19x19
                            max_pool<2,2,2,2,prelu<bn_con<con<16,10,10,1,1,     // 82x82 -> 41x41
                            prelu<bn_con<con<8,20,20,2,2,                       // 91x91
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>;

using eye_bow_net = loss_mean_squared_multioutput<
                            fc<10,
                            prelu<bn_con<con<32,3,3,1,1,                        // 13x5
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 29x14 -> 15x7
                            max_pool<2,2,2,2,prelu<bn_con<con<16,20,20,1,1,     // 80x50 -> 31x16
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>;
using eye_net = loss_mean_squared_multioutput<
                            fc<12,
                            prelu<bn_con<con<32,3,3,1,1,                        // 10x5
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 23x14 -> 12x7
                            max_pool<2,2,2,2,prelu<bn_con<con<16,20,20,1,1,     // 70x50 -> 25x16
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>;
using nose_net = loss_mean_squared_multioutput<
                            fc<18,
                            prelu<bn_con<con<32,3,3,1,1,                        // 10x15
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 23x33 -> 15x17
                            max_pool<2,2,2,2,prelu<bn_con<con<16,20,20,1,1,     // 70x90 -> 25x35
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>;

// параметры при запуске программы
void setParams(po::variables_map &vm)
{
    patches_sizes = 80, 80, 70, 70, 70,
                    50, 50, 90, 50, 50;
    points_sizes = 5, 5, 9, 6, 6;

    if(vm.count("nn"))
    {
        network_name = vm["nn"].as<string>();
        cout << "Neural network name is: " << network_name << endl;
    }

    if(vm.count("in"))
    {
        iter_name = vm["in"].as<string>();
        cout << "Iteration number is: " << iter_name << endl;
    }

    if(vm.count("sd"))
    {
        snapshot_directory_path = vm["sd"].as<string>();
        cout << "Snapshot folder path is: " << snapshot_directory_path << endl;
    }
    if(vm.count("m_num"))
    {
        multiply_network_number = vm["m_num"].as<int>();
        cout << "Multiply network number: " << multiply_network_number << endl;
    }
    if(vm.count("vp"))
    {
        video_path = vm["vp"].as<string>();
        cout << "Video folder path is: " << video_path << endl;
    }
    if(vm.count("cascade"))
    {
        cascade_num = vm["cascade"].as<int>();
        cout << "Cascade number: " << cascade_num << endl;
    }

    if(vm.count("draftnn"))
    {
        draft_network_name = vm["draftnn"].as<string>();
        cout << "Draft neural network name is: " << draft_network_name << endl;
    }

    if(vm.count("draftin"))
    {
        draft_iter_name = vm["draftin"].as<string>();
        cout << "Draft iteration number is: " << draft_iter_name << endl;
    }

    if(vm.count("net_leb_in"))
    {
        iter_name_leb = vm["net_leb_in"].as<string>();
        cout << "Left eyebow iteration number is: " << iter_name_leb << endl;
    }
    if(vm.count("net_leb_nn"))
    {
        network_name_leb = vm["net_leb_nn"].as<string>();
        cout << "Left eyebow neural network name is: " << network_name_leb << endl;
    }

    if(vm.count("net_reb_in"))
    {
        iter_name_reb = vm["net_reb_in"].as<string>();
        cout << "Right eyebow iteration number is: " << iter_name_reb << endl;
    }
    if(vm.count("net_reb_nn"))
    {
        network_name_reb = vm["net_reb_nn"].as<string>();
        cout << "Right eyebow neural network name is: " << network_name_reb << endl;
    }

    if(vm.count("net_nose_in"))
    {
        iter_name_nose = vm["net_nose_in"].as<string>();
        cout << "Nose iteration number is: " << iter_name_nose << endl;
    }

    if(vm.count("net_nose_nn"))
    {
        network_name_nose = vm["net_nose_nn"].as<string>();
        cout << "Nose neural network name is: " << network_name_nose << endl;
    }

    if(vm.count("net_le_in"))
    {
        iter_name_le = vm["net_le_in"].as<string>();
        cout << "Left eye iteration number is: " << iter_name_le << endl;
    }
    if(vm.count("net_le_nn"))
    {
        network_name_le = vm["net_le_nn"].as<string>();
        cout << "Left eye neural network name is: " << network_name_le << endl;
    }

    if(vm.count("net_re_in"))
    {
        iter_name_re = vm["net_re_in"].as<string>();
        cout << "Right eye iteration number is: " << iter_name_re << endl;
    }
    if(vm.count("net_re_nn"))
    {
        network_name_re = vm["net_re_nn"].as<string>();
        cout << "Right eye neural network name is: " << network_name_re << endl;
    }
}

// функция для запуска видео/вебкамеры
template<class T>
void test(T &net)
{
    // захват  камеры
    cv::VideoCapture capture(0); // "/home/ginseng/Projects/DataSet/test.avi" / "/home/ginseng/Projects/DataSet/Surgut.avi"

    // фрейм с камеры
    cv::Mat frame, newframe, newframerotated;

    // матрица dliba
    matrix<uchar> faceRectDlib, frameDlib, faceRectDlibRotated;

    // точки
    dlib::matrix<float> testNN, testNNRotated;

    // окно
    dlib::image_window win, win2, win3;

    // точка
    dlib::point po, pod;

    // детектор лиц
    CpuFaceDetector FaceDetect("haarcascade_frontalface_alt.xml", 90);
    std::vector<CpuFaceDetector::FaceInfo> faces;

    // подключение сети
    string NetName = snapshot_directory_path + "/" + network_name + "/" + network_name + "_iter_" + iter_name + ".dat";
    deserialize(NetName) >> net;

    int rot = 0;

    // dlib shape predictor
    shape_predictor sp, spr;
    deserialize("test/shape_predictor_dlib.dat") >> sp;
    deserialize("test/shape_predictor_dlib.dat") >> spr;
    full_object_detection shape, shape_rot;

    while(true)
    {
        // получаем кадр
        capture.read(frame);

        // детектирование
        faces = FaceDetect.detect(frame);

        assign_image(frameDlib, cv_image<bgr_pixel>(frame));
        win.set_image(frameDlib);

        if (faces.size() != 0)
        {
            // несколько лиц на изображении
            // устанавливаем фрейм в созданое окно и очищаем лейаут
            win.clear_overlay();
            for (int i = 0; i < faces.size(); ++i)
            {
                win2.clear_overlay();
                win3.clear_overlay();
                // находим само лицо на изображении
                newframe = frame(faces[i].rect);

                // преобразуем в градации серого
                cv::cvtColor(newframe, newframe, CV_BGR2GRAY);
                cv::resize(newframe,newframe,cv::Size(200,200));

                // преобразуем в длиб для большого и отдельных окон
                assign_image(faceRectDlib, cv_image<uchar>(newframe));

                // вращаем
                cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(newframe.cols/2, newframe.rows/2), rot, 1);
                cv::warpAffine(newframe, newframerotated, M, cv::Size(newframe.cols,newframe.rows));
                assign_image(faceRectDlibRotated, cv_image<uchar>(newframerotated));

                // посылаем сети и длибу изображение с поворотом
                testNNRotated = net(faceRectDlibRotated);
                shape_rot = spr(faceRectDlibRotated, get_rect(faceRectDlibRotated));

                // засекаем время выполнения
                boost::posix_time::ptime testTimemcs = boost::posix_time::microsec_clock::universal_time();
                testNN = net(faceRectDlib);
                auto time = (boost::posix_time::microsec_clock::universal_time() - testTimemcs).total_microseconds();
                cout << "Time per frame: " << time << " mcs." << endl;

                // отправляем также в длиб
                shape = sp(faceRectDlib, get_rect(faceRectDlib));

                win.add_overlay(dlib::rectangle((long)faces[i].rect.tl().x, (long)faces[i].rect.tl().y, (long)faces[i].rect.br().x - 1, (long)faces[i].rect.br().y - 1), rgb_pixel(0,255,0));
                win2.set_image(faceRectDlibRotated);
                win3.set_image(faceRectDlibRotated);

                // расставляем точки
                for (int j = 0; j < 31; ++j)
                {
                    // добавляем точки на фрейм
                    po.x() = ((testNN(j*2)*faces[i].rect.width)/200) + faces[i].rect.x; //*200)/faces[i].rect.width)
                    po.y() = ((testNN(j*2+1)*faces[i].rect.height)/200) + faces[i].rect.y; // *200)/faces[i].rect.height)
                    win.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win.add_overlay(po, po, rgb_pixel(255,0,0));
                    pod.x() = ((shape.part(j + 17).x()*faces[i].rect.width)/200) + faces[i].rect.x;
                    pod.y() = ((shape.part(j + 17).y()*faces[i].rect.height)/200) + faces[i].rect.y;
                    win.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win.add_overlay(pod, pod, rgb_pixel(0,0,255));
                    // добавляем точки на окно
                    po.x() = testNNRotated(j*2);
                    po.y() = testNNRotated(j*2+1);
                    win2.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win2.add_overlay(po, po, rgb_pixel(255,0,0));
                    pod.x() = shape_rot.part(j + 17).x();
                    pod.y() = shape_rot.part(j + 17).y();
                    win3.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win3.add_overlay(pod, pod, rgb_pixel(0,0,255));
                }

            }
        }
        /*usleep(50000);
        rot += 5;
        if (rot > 25) rot = -25;*/
    //cin.get();

    }
}
// функция для проверки изображений
template<class T>
void draw(T &net)
{
    std::string newpath; // путь

    QString dirpath = QString::fromStdString(webcam_parametre);

    // фрейм
    cv::Mat frame, newframe, framebgr;

    // матрица dliba
    matrix<uchar> faceRectDlib;
    dlib::array2d<bgr_pixel> frameDlib;

    // точки
    dlib::matrix<float> testNN;

    // окно
    dlib::image_window win, win2;

    // точка
    dlib::point po, pod;
    cv::Point2f cvpo, cvpod;

    // детектор лиц
    CpuFaceDetector FaceDetect("haarcascade_frontalface_alt.xml", 90);
    std::vector<CpuFaceDetector::FaceInfo> faces;

    // подключение сети
    string NetName = snapshot_directory_path + "/" + network_name + "/" + network_name + "_iter_" + iter_name + ".dat";
    deserialize(NetName) >> net;

    // dlib shape predictor
    shape_predictor sp;
    deserialize("test/shape_predictor_dlib.dat") >> sp;
    full_object_detection shape;

    QDirIterator it(dirpath, QStringList() << "*.jpg", QDir::Files, QDirIterator::Subdirectories);

    while(it.hasNext())
    {
        it.next();

        frame = imread(it.filePath().toStdString(), CV_LOAD_IMAGE_UNCHANGED);
        framebgr = frame.clone();
        faces.clear();

        // детектирование
        faces = FaceDetect.detect(frame);

        assign_image(frameDlib, cv_image<bgr_pixel>(framebgr));
        win.set_image(frameDlib);

        if (faces.size() != 0)
        {
            // устанавливаем фрейм в созданое окно и очищаем лейаут
            win.clear_overlay();
            win2.clear_overlay();
            for (int i = 0; i < faces.size(); ++i)
            {
                // находим само лицо на изображении
                newframe = frame(faces[i].rect);

                // преобразуем в градации серого
                cv::cvtColor(newframe, newframe, CV_BGR2GRAY);
                cv::resize(newframe,newframe,cv::Size(200,200));

                // преобразуем в длиб для большого и отдельных окон
                assign_image(faceRectDlib, cv_image<uchar>(newframe));

                // засекаем время выполнения
                boost::posix_time::ptime testTimemcs = boost::posix_time::microsec_clock::universal_time();
                testNN = net(faceRectDlib);
                auto time = (boost::posix_time::microsec_clock::universal_time() - testTimemcs).total_microseconds();
                cout << "Time per frame: " << time << " mcs." << endl;

                // отправляем также в длиб
                shape = sp(faceRectDlib, get_rect(faceRectDlib));

                win.add_overlay(dlib::rectangle((long)faces[i].rect.tl().x, (long)faces[i].rect.tl().y, (long)faces[i].rect.br().x - 1, (long)faces[i].rect.br().y - 1), rgb_pixel(0,255,0));
                win2.set_image(faceRectDlib);
                // расставляем точки
                for (int j = 0; j < 31; ++j)
                {
                    // добавляем точки на фрейм
                    po.x() = ((testNN(j*2)*faces[i].rect.width)/200) + faces[i].rect.x; //*200)/faces[i].rect.width)
                    po.y() = ((testNN(j*2+1)*faces[i].rect.height)/200) + faces[i].rect.y; // *200)/faces[i].rect.height)
                    win.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win.add_overlay(po, po, rgb_pixel(255,0,0));
                    cvpo.x = ((testNN(j*2)*faces[i].rect.width)/200) + faces[i].rect.x;
                    cvpo.y = ((testNN(j*2+1)*faces[i].rect.height)/200) + faces[i].rect.y;
                    cv::circle(framebgr,cvpo,2,CV_RGB(255,0,0),1,8,0);


                    pod.x() = ((shape.part(j + 17).x()*faces[i].rect.width)/200) + faces[i].rect.x;
                    pod.y() = ((shape.part(j + 17).y()*faces[i].rect.height)/200) + faces[i].rect.y;
                    win.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win.add_overlay(pod, pod, rgb_pixel(0,0,255));
                    cvpod.x = ((shape.part(j + 17).x()*faces[i].rect.width)/200) + faces[i].rect.x;
                    cvpod.y = ((shape.part(j + 17).y()*faces[i].rect.height)/200) + faces[i].rect.y;
                    cv::circle(framebgr,cvpod,2,CV_RGB(0,0,255),1,8,0);

                    // добавляем точки на фрейм
                    po.x() = testNN(j*2);
                    po.y() = testNN(j*2+1);
                    win2.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win2.add_overlay(po, po, rgb_pixel(255,0,0));


                    pod.x() = shape.part(j + 17).x();
                    pod.y() = shape.part(j + 17).y();
                    win2.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win2.add_overlay(pod, pod, rgb_pixel(0,0,255));
                }

                newpath = "/home/ginseng/BaseForTest/res/" + it.fileInfo().baseName().toStdString() + "_T" + to_string(i) + ".png";
                //imwrite(newpath, framebgr);

            }
        }
    cin.get();

    }
}


// вспомогательная функция для вырезки фрагмента изображения
dlib::matrix<uchar> get_patch(cv::Mat img, std::vector<float> &coords, dlib::matrix<float> &train, int patch_size_x, int patch_size_y, int part, int num)
{
    dlib::matrix<uchar> ans;
    dlib::point minp, maxp;
    cv::Mat patch;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < num; ++j)
    {
        if (train(j*2+part) < minp.x()) minp.x() = train(j*2+part);
        if (train(j*2+part) > maxp.x()) maxp.x() = train(j*2+part);
        if (train(j*2+part+1) < minp.y()) minp.y() = train(j*2+part+1);
        if (train(j*2+part+1) > maxp.y()) maxp.y() = train(j*2+part+1);
    }
    // добавляем несколько пикселов с учётом границ изображения
    int desc = 15;
    minp.x() -= desc;
    while (minp.x() < 0)
    {
        minp.x()++;
    }
    minp.y() -= desc;
    while (minp.y() < 0)
    {
        minp.y()++;
    }
    maxp.x() += desc;
    while (maxp.x() > img.cols)
    {
        maxp.x()--;
    }
    maxp.y() += desc;
    while (maxp.y() > img.rows)
    {
        maxp.y()--;
    }
    // сохраняем результаты в
    coords.push_back(minp.x()); coords.push_back(minp.y()); coords.push_back(maxp.x()); coords.push_back(maxp.y());

    // меняем размер и координаты точек
    patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
    cv::resize(patch, patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);

    assign_image(ans, cv_image<uchar>(patch));
    return ans;
}
// функиця восстановление точек из фрагментов на начальное изображение
dlib::matrix<float> restore_image(dlib::matrix<float> &testNN, dlib::matrix<float> &leb_patch, dlib::matrix<float> &reb_patch, dlib::matrix<float> &n_patch, dlib::matrix<float> &le_patch, dlib::matrix<float> &re_patch, std::vector<float> &coords)
{
    dlib::matrix<float> answer;
    answer.set_size(testNN.nr(), testNN.nc());

    int bias = 0;
    int coord_bias = 0;
    // Левая бровь
    if (leb_patch(0) == -1)
    {
        for (int i = 0; i < points_sizes(0,0); ++i)
        {
            answer(i * 2 + bias) = testNN(i * 2 + bias);
            answer(i * 2 + 1 + bias) = testNN(i * 2 + bias + 1);
        }
    }
    else
    {
        for (int i = 0; i < points_sizes(0,0); ++i)
        {
            answer(i * 2 + bias) = leb_patch(i * 2) * (coords[2 + coord_bias] - coords[coord_bias]) / patches_sizes(0, 0) + coords[coord_bias];
            answer(i * 2 + 1 + bias) = leb_patch(i * 2 + 1) * (coords[3 + coord_bias] - coords[1 + coord_bias]) / patches_sizes(1, 0) + coords[1 + coord_bias];
        }
        coord_bias += 4;
    }

    bias += points_sizes(0,0)*2;
    // Правая бровь
    if (reb_patch(0) == -1)
    {
        for (int i = 0; i < points_sizes(0,1); ++i)
        {
            answer(i * 2 + bias) = testNN(i * 2 + bias);
            answer(i * 2 + 1 + bias) = testNN(i * 2 + bias + 1);
        }
    }
    else
    {
        for (int i = 0; i < points_sizes(0,1); ++i)
        {
            answer(i * 2 + bias) = reb_patch(i * 2) * (coords[2 + coord_bias] - coords[coord_bias]) / patches_sizes(0, 1) + coords[coord_bias];
            answer(i * 2 + 1 + bias) = reb_patch(i * 2 + 1) * (coords[3 + coord_bias] - coords[1 + coord_bias]) / patches_sizes(1, 1) + coords[1 + coord_bias];
        }
        coord_bias += 4;
    }
    bias += points_sizes(0,1)*2;
    // Нос
    if (n_patch(0) == -1)
    {
        for (int i = 0; i < points_sizes(0,2); ++i)
        {
            answer(i * 2 + bias) = testNN(i * 2 + bias);
            answer(i * 2 + 1 + bias) = testNN(i * 2 + bias + 1);
        }
    }
    else
    {
        for (int i = 0; i < points_sizes(0,2); ++i)
        {
            answer(i * 2 + bias) = n_patch(i * 2) * (coords[2 + coord_bias] - coords[coord_bias]) / patches_sizes(0, 2) + coords[coord_bias];
            answer(i * 2 + 1 + bias) = n_patch(i * 2 + 1) * (coords[3 + coord_bias] - coords[1 + coord_bias]) / patches_sizes(1, 2) + coords[1 + coord_bias];
        }
        coord_bias += 4;
    }
    bias += points_sizes(0,2)*2;
    // Левый глаз
    if (le_patch(0) == -1)
    {
        for (int i = 0; i < points_sizes(0,3); ++i)
        {
            answer(i * 2 + bias) = testNN(i * 2 + bias);
            answer(i * 2 + 1 + bias) = testNN(i * 2 + bias + 1);
        }
    }
    else
    {
        for (int i = 0; i < points_sizes(0,3); ++i)
        {
            answer(i * 2 + bias) = le_patch(i * 2) * (coords[2 + coord_bias] - coords[coord_bias]) / patches_sizes(0, 3) + coords[coord_bias];
            answer(i * 2 + 1 + bias) = le_patch(i * 2 + 1) * (coords[3 + coord_bias] - coords[1 + coord_bias]) / patches_sizes(1, 3) + coords[1 + coord_bias];
        }
        coord_bias += 4;
    }
    bias += points_sizes(0,3)*2;
    // Правый глаз
    if (re_patch(0) == -1)
    {
        for (int i = 0; i < points_sizes(0,4); ++i)
        {
            answer(i * 2 + bias) = testNN(i * 2 + bias);
            answer(i * 2 + 1 + bias) = testNN(i * 2 + bias + 1);
        }
    }
    else
    {
        for (int i = 0; i < points_sizes(0,4); ++i)
        {
            answer(i * 2 + bias) = re_patch(i * 2) * (coords[2 + coord_bias] - coords[coord_bias]) / patches_sizes(0, 4) + coords[coord_bias];
            answer(i * 2 + 1 + bias) = re_patch(i * 2 + 1) * (coords[3 + coord_bias] - coords[1 + coord_bias]) / patches_sizes(1, 4) + coords[1 + coord_bias];
        }
    }
    //cin.get();
    return answer;
}
// перегруженные функции для запуска проверки по видео или изображениям
template<class T1, class T2, class T3>
void test(T1 &leb_net, T1 &reb_net, T2 &n_net, T3 &le_net, T3 &re_net)
{
    // захват  камеры
    cv::VideoCapture capture(0); // "/home/ginseng/Projects/DataSet/test.avi" / "/home/ginseng/Projects/DataSet/Surgut.avi"

    // фрейм с камеры
    cv::Mat frame, newframe;

    // матрица dliba
    matrix<uchar> faceRectDlib, frameDlib;

    // точки
    dlib::matrix<float> testNN;

    // окно
    dlib::image_window win, win2, win3;

    // точка
    dlib::point po, pod;

    // детектор лиц
    CpuFaceDetector FaceDetect("haarcascade_frontalface_alt.xml", 90);
    std::vector<CpuFaceDetector::FaceInfo> faces;

    draft_net_type Draft_Net;
    string DraftNetName = snapshot_directory_path + "/" + draft_network_name + "/" + draft_network_name + "_iter_" + draft_iter_name + ".dat";
    // загружаем полученную сеть
    deserialize(DraftNetName) >> Draft_Net;

    // подключение сети левой брови
    string LeftEyeBowNetName = snapshot_directory_path + "/" + network_name_leb + "/" + network_name_leb + "_iter_" + iter_name_leb + ".dat";
    deserialize(LeftEyeBowNetName) >> leb_net;

    // подключение сети правой брови
    string RightEyeBowNetName = snapshot_directory_path + "/" + network_name_reb + "/" + network_name_reb + "_iter_" + iter_name_reb + ".dat";
    deserialize(RightEyeBowNetName) >> reb_net;

    // подключение сети носа
    string NoseNetName = snapshot_directory_path + "/" + network_name_nose + "/" + network_name_nose + "_iter_" + iter_name_nose + ".dat";
    deserialize(NoseNetName) >> n_net;

    // подключение сети левого глаза
    string LeftEyeNetName = snapshot_directory_path + "/" + network_name_le + "/" + network_name_le + "_iter_" + iter_name_le + ".dat";
    deserialize(LeftEyeNetName) >> le_net;

    // подключение сети правого глаза
    string RightEyeNetName = snapshot_directory_path + "/" + network_name_re + "/" + network_name_re + "_iter_" + iter_name_re + ".dat";
    deserialize(RightEyeNetName) >> re_net;

    dlib::matrix<float> leb_patch, reb_patch, n_patch, le_patch, re_patch, restored;


    // dlib shape predictor
    shape_predictor sp;
    deserialize("test/shape_predictor_dlib.dat") >> sp;
    full_object_detection shape;

    std::vector<float> coords;


    while(true)
    {
        // получаем кадр
        capture.read(frame);

        // детектирование
        faces = FaceDetect.detect(frame);

        assign_image(frameDlib, cv_image<bgr_pixel>(frame));
        win.set_image(frameDlib);

        if (faces.size() != 0)
        {
            // несколько лиц на изображении
            // устанавливаем фрейм в созданое окно и очищаем лейаут
            win.clear_overlay();
            win2.clear_overlay();
            win3.clear_overlay();

            for (int i = 0; i < faces.size(); ++i)
            {
                coords.clear();

                // находим само лицо на изображении
                newframe = frame(faces[i].rect);

                // преобразуем в градации серого
                cv::cvtColor(newframe, newframe, CV_BGR2GRAY);
                cv::resize(newframe,newframe,cv::Size(200,200));

                // преобразуем в длиб для большого и отдельных окон
                assign_image(faceRectDlib, cv_image<uchar>(newframe));

                // засекаем время выполнения
                boost::posix_time::ptime testTimemcs = boost::posix_time::microsec_clock::universal_time();

                // --- MEMO ---

                boost::posix_time::ptime DraftTimemcs = boost::posix_time::microsec_clock::universal_time();

                // находим черновые точки
                testNN = Draft_Net(faceRectDlib);

                auto time1 = (boost::posix_time::microsec_clock::universal_time() - DraftTimemcs).total_microseconds();
                cout << "Draft NN: " << time1 << " mcs." << endl;

                // по ним вырезаем части изображения и ищем на них точки более точно

                /*boost::posix_time::ptime LebTimemcs = boost::posix_time::microsec_clock::universal_time();

                // для левой брови
                leb_patch = leb_net(get_patch(newframe, coords, testNN, patches_sizes(0,0), patches_sizes(1,0), 0, points_sizes(0,0)));

                auto time2 = (boost::posix_time::microsec_clock::universal_time() - LebTimemcs).total_microseconds();
                cout << "LEB NN: " << time2 << " mcs." << endl;*/

                /*boost::posix_time::ptime RebTimemcs = boost::posix_time::microsec_clock::universal_time();

                // для правой брови
                reb_patch = reb_net(get_patch(newframe, coords, testNN, patches_sizes(0,1), patches_sizes(1,1), 10, points_sizes(0,1)));

                auto time3 = (boost::posix_time::microsec_clock::universal_time() - RebTimemcs).total_microseconds();
                cout << "REB NN: " << time3 << " mcs." << endl;*/

                /*boost::posix_time::ptime NoseTimemcs = boost::posix_time::microsec_clock::universal_time();

                // для носа
                n_patch = n_net(get_patch(newframe, coords, testNN, patches_sizes(0,2), patches_sizes(1,2), 20, points_sizes(0,2)));

                auto time4 = (boost::posix_time::microsec_clock::universal_time() - NoseTimemcs).total_microseconds();
                cout << "NOSE NN: " << time4 << " mcs." << endl;*/

                boost::posix_time::ptime LeTimemcs = boost::posix_time::microsec_clock::universal_time();

                // для левого глаза
                le_patch = le_net(get_patch(newframe, coords, testNN, patches_sizes(0,3), patches_sizes(1,3), 38, points_sizes(0,3)));

                auto time5 = (boost::posix_time::microsec_clock::universal_time() - LeTimemcs).total_microseconds();
                cout << "LE NN: " << time5 << " mcs." << endl;

                boost::posix_time::ptime ReTimemcs = boost::posix_time::microsec_clock::universal_time();

                // для правого глаза
                re_patch = re_net(get_patch(newframe, coords, testNN, patches_sizes(0,4), patches_sizes(1,4), 50, points_sizes(0,4)));

                auto time6 = (boost::posix_time::microsec_clock::universal_time() - ReTimemcs).total_microseconds();
                cout << "RE NN: " << time6 << " mcs." << endl;

                boost::posix_time::ptime RestTimemcs = boost::posix_time::microsec_clock::universal_time();

                // функция для сбора точек на изображение
                restored.set_size(testNN.nr(), testNN.nc());
                dlib::matrix<float> t(1,1);
                t = -1;
                restored = restore_image(testNN, t, t, t, le_patch, re_patch, coords);

                auto time7 = (boost::posix_time::microsec_clock::universal_time() - RestTimemcs).total_microseconds();
                cout << "RESTORE: " << time7 << " mcs." << endl;

                auto time = (boost::posix_time::microsec_clock::universal_time() - testTimemcs).total_microseconds();
                cout << "Neural network time per frame: " << time << " mcs." << endl;

                // отправляем также в длиб
                boost::posix_time::ptime dlibTimemcs = boost::posix_time::microsec_clock::universal_time();
                shape = sp(faceRectDlib, get_rect(faceRectDlib));
                auto dlibtime = (boost::posix_time::microsec_clock::universal_time() - dlibTimemcs).total_microseconds();
                cout << "Dlib time per frame: " << dlibtime << " mcs." << endl << "Comp: " << 100 * time / dlibtime << "%" << endl;

                win.add_overlay(dlib::rectangle((long)faces[i].rect.tl().x, (long)faces[i].rect.tl().y, (long)faces[i].rect.br().x - 1, (long)faces[i].rect.br().y - 1), rgb_pixel(0,255,0));
                win2.set_image(faceRectDlib);
                win3.set_image(faceRectDlib);
                win2.clear_overlay();
                win3.clear_overlay();
                // расставляем точки
                for (int j = 0; j < 31; ++j)
                {
                    // добавляем точки на фрейм
                    po.x() = ((restored(j*2)*faces[i].rect.width)/200) + faces[i].rect.x; //*200)/faces[i].rect.width)
                    po.y() = ((restored(j*2+1)*faces[i].rect.height)/200) + faces[i].rect.y; // *200)/faces[i].rect.height)
                    win.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win.add_overlay(po, po, rgb_pixel(255,0,0));
                    pod.x() = ((shape.part(j + 17).x()*faces[i].rect.width)/200) + faces[i].rect.x;
                    pod.y() = ((shape.part(j + 17).y()*faces[i].rect.height)/200) + faces[i].rect.y;
                    win.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win.add_overlay(pod, pod, rgb_pixel(0,0,255));
                    // добавляем точки на окно
                    po.x() = restored(j*2);
                    po.y() = restored(j*2+1);
                    win2.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                    win2.add_overlay(po, po, rgb_pixel(255,0,0));
                    pod.x() = shape.part(j + 17).x();
                    pod.y() = shape.part(j + 17).y();
                    win3.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                    win3.add_overlay(pod, pod, rgb_pixel(0,0,255));
                }

            }
        }

        /*usleep(50000);
        rot += 5;
        if (rot > 25) rot = -25;*/
    //cin.get();

    }
}
template<class T1, class T2, class T3>
void draw(T1 &leb_net, T1 &reb_net, T2 &n_net, T3 &le_net, T3 &re_net)
{
    if (typeid(webcam_parametre).name() == "string")
    {
        QString dirpath = QString::fromStdString(webcam_parametre);

        // фрейм
        cv::Mat frame, newframe;

        // матрица dliba
        matrix<uchar> faceRectDlib, frameDlib;

        // точки
        dlib::matrix<float> testNN;

        // окно
        dlib::image_window win, win2, win3;

        // точка
        dlib::point po, pod;

        // детектор лиц
        CpuFaceDetector FaceDetect("haarcascade_frontalface_alt.xml", 90);
        std::vector<CpuFaceDetector::FaceInfo> faces;

        draft_net_type Draft_Net;
        string DraftNetName = snapshot_directory_path + "/" + draft_network_name + "/" + draft_network_name + "_iter_" + draft_iter_name + ".dat";
        // загружаем полученную сеть
        deserialize(DraftNetName) >> Draft_Net;

        // подключение сети левой брови
        string LeftEyeBowNetName = snapshot_directory_path + "/" + network_name_leb + "/" + network_name_leb + "_iter_" + iter_name_leb + ".dat";
        deserialize(LeftEyeBowNetName) >> leb_net;

        // подключение сети правой брови
        string RightEyeBowNetName = snapshot_directory_path + "/" + network_name_reb + "/" + network_name_reb + "_iter_" + iter_name_reb + ".dat";
        deserialize(RightEyeBowNetName) >> reb_net;

        // подключение сети носа
        string NoseNetName = snapshot_directory_path + "/" + network_name_nose + "/" + network_name_nose + "_iter_" + iter_name_nose + ".dat";
        deserialize(NoseNetName) >> n_net;

        // подключение сети левого глаза
        string LeftEyeNetName = snapshot_directory_path + "/" + network_name_le + "/" + network_name_le + "_iter_" + iter_name_le + ".dat";
        deserialize(LeftEyeNetName) >> le_net;

        // подключение сети правого глаза
        string RightEyeNetName = snapshot_directory_path + "/" + network_name_re + "/" + network_name_re + "_iter_" + iter_name_re + ".dat";
        deserialize(RightEyeNetName) >> re_net;

        dlib::matrix<float> leb_patch, reb_patch, n_patch, le_patch, re_patch, restored;
        cv::Mat patch;


        // dlib shape predictor
        shape_predictor sp;
        deserialize("test/shape_predictor_dlib.dat") >> sp;
        full_object_detection shape;

        std::vector<float> coords;

        QDirIterator it(dirpath, QStringList() << "*.jpg", QDir::Files, QDirIterator::Subdirectories);

        while(it.hasNext())
        {

            it.next();

            frame = imread(it.filePath().toStdString(), CV_LOAD_IMAGE_UNCHANGED);
            faces.clear();

            // детектирование
            faces = FaceDetect.detect(frame);

            assign_image(frameDlib, cv_image<bgr_pixel>(frame));
            win.set_image(frameDlib);

            if (faces.size() != 0)
            {
                // несколько лиц на изображении
                // устанавливаем фрейм в созданое окно и очищаем лейаут
                win.clear_overlay();
                win2.clear_overlay();
                win3.clear_overlay();

                for (int i = 0; i < faces.size(); ++i)
                {
                    coords.clear();

                    // находим само лицо на изображении
                    newframe = frame(faces[i].rect);

                    // преобразуем в градации серого
                    cv::cvtColor(newframe, newframe, CV_BGR2GRAY);
                    cv::resize(newframe,newframe,cv::Size(200,200));

                    // преобразуем в длиб для большого и отдельных окон
                    assign_image(faceRectDlib, cv_image<uchar>(newframe));

                    // засекаем время выполнения
                    boost::posix_time::ptime testTimemcs = boost::posix_time::microsec_clock::universal_time();

                    // находим черновые точки
                    testNN = Draft_Net(faceRectDlib);
                    // по ним вырезаем части изображения и ищем на них точки более точно
                    // для левой брови
                    leb_patch = leb_net(get_patch(newframe, coords, testNN, 80, 50, 0, 5));
                    // для правой брови
                    reb_patch = reb_net(get_patch(newframe, coords, testNN, 80, 50, 10, 5));
                    // для носа
                    n_patch = n_net(get_patch(newframe, coords, testNN, 70, 90, 20, 9));
                    // для левого глаза
                    le_patch = le_net(get_patch(newframe, coords, testNN, 70, 50, 38, 6));
                    // для правого глаза
                    re_patch = re_net(get_patch(newframe, coords, testNN, 70, 50, 50, 6));
                    // функция для сбора точек на изображение
                    restored = restore_image(testNN, leb_patch, reb_patch, n_patch, le_patch, re_patch, coords);

                    auto time = (boost::posix_time::microsec_clock::universal_time() - testTimemcs).total_microseconds();
                    cout << "Neural network time per frame: " << time << " mcs." << endl;

                    // отправляем также в длиб
                    boost::posix_time::ptime dlibTimemcs = boost::posix_time::microsec_clock::universal_time();
                    shape = sp(faceRectDlib, get_rect(faceRectDlib));
                    auto dlibtime = (boost::posix_time::microsec_clock::universal_time() - dlibTimemcs).total_microseconds();
                    cout << "Dlib time per frame: " << dlibtime << " mcs." << endl << "Comp: " << time / (time + dlibtime) << "%" << endl;

                    win.add_overlay(dlib::rectangle((long)faces[i].rect.tl().x, (long)faces[i].rect.tl().y, (long)faces[i].rect.br().x - 1, (long)faces[i].rect.br().y - 1), rgb_pixel(0,255,0));
                    win2.set_image(faceRectDlib);
                    win3.set_image(faceRectDlib);
                    // расставляем точки
                    for (int j = 0; j < 31; ++j)
                    {
                        // добавляем точки на фрейм
                        po.x() = ((restored(j*2)*faces[i].rect.width)/200) + faces[i].rect.x; //*200)/faces[i].rect.width)
                        po.y() = ((restored(j*2+1)*faces[i].rect.height)/200) + faces[i].rect.y; // *200)/faces[i].rect.height)
                        win.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                        win.add_overlay(po, po, rgb_pixel(255,0,0));
                        pod.x() = ((shape.part(j + 17).x()*faces[i].rect.width)/200) + faces[i].rect.x;
                        pod.y() = ((shape.part(j + 17).y()*faces[i].rect.height)/200) + faces[i].rect.y;
                        win.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                        win.add_overlay(pod, pod, rgb_pixel(0,0,255));
                        // добавляем точки на окно
                        po.x() = restored(j*2);
                        po.y() = restored(j*2+1);
                        win2.add_overlay(image_window::overlay_circle(po, 2, rgb_pixel(255,0,0)));
                        win2.add_overlay(po, po, rgb_pixel(255,0,0));
                        pod.x() = shape.part(j + 17).x();
                        pod.y() = shape.part(j + 17).y();
                        win3.add_overlay(image_window::overlay_circle(pod, 2, rgb_pixel(0,0,255)));
                        win3.add_overlay(pod, pod, rgb_pixel(0,0,255));
                    }

                }
            }

            /*usleep(50000);
            rot += 5;
            if (rot > 25) rot = -25;*/
        //cin.get();

        }
    }
}

// оснровная функиция программы
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    po::options_description desc("Application options");
    desc.add_options()
        ("help", "Show help message")
        ("nn", po::value<std::string>()->required()->default_value("Test"), "Set network name")
        ("sd", po::value<std::string>()->required()->default_value("/home/ginseng/Projects/DataSet/Snapshots"), "Set snapshot directory")
        ("in", po::value<std::string>()->default_value("10000"), "Set snapshot iteration number")
        ("m_num", po::value<int>()->default_value(0), "Set number of multiply network")
        ("vp", po::value<std::string>()->default_value("0"), "Set video folder")
        ("cascade", po::value<int>()->required()->default_value(0), "Set cascade number")
        ("draftnn", po::value<std::string>(), "Set draft network name")
        ("draftin", po::value<std::string>()->default_value("10000"), "Set draft snapshot iteration number")
        ("net_leb_nn", po::value<std::string>(), "Set left eyebow network name")
        ("net_leb_in", po::value<std::string>()->default_value("10000"), "Set left eyebow snapshot iteration number")
        ("net_reb_nn", po::value<std::string>(), "Set right eyebow network name")
        ("net_reb_in", po::value<std::string>()->default_value("10000"), "Set right eyebow snapshot iteration number")
        ("net_nose_nn", po::value<std::string>(), "Set nose network name")
        ("net_nose_in", po::value<std::string>()->default_value("10000"), "Set nose snapshot iteration number")
        ("net_le_nn", po::value<std::string>(), "Set left eye network name")
        ("net_le_in", po::value<std::string>()->default_value("10000"), "Set left eye snapshot iteration number")
        ("net_re_nn", po::value<std::string>(), "Set right eye network name")
        ("net_re_in", po::value<std::string>()->default_value("10000"), "Set right eye snapshot iteration number")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    setParams(vm);
    std::cout << std::endl;

    char answer;
    cout << "Check videos (V) or images (I)" << endl;
    cin >> answer;
    // если выбрана проверка каскада
    if (cascade_num > 0)
    {
        eye_bow_net leb_net;
        eye_bow_net reb_net;
        nose_net n_net;
        eye_net le_net;
        eye_net re_net;

        if (answer == 'V') test(leb_net, reb_net, n_net, le_net, re_net);
        if (answer == 'I') draw(leb_net, reb_net, n_net, le_net, re_net);
    }
    else // иначе проверяем одну из цельных сетей
    switch (multiply_network_number)
    {
    case 0:
        {
            // объявление сети
            net_type net;

            if (answer == 'V') test(net);
            if (answer == 'I') draw(net);

            break;
        }
    case 1:
        {
            // объявление сети
            net_type_2 net;

            if (answer == 'V') test(net);
            if (answer == 'I') draw(net);

            break;
        }
    case 2:
        {
            // объявление сети
            net_type_3 net;

            if (answer == 'V') test(net);
            if (answer == 'I') draw(net);

            break;
        }
    }

    return a.exec();
}
