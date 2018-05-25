#include <QCoreApplication>

#include "LMDBwrapper/LMDB.h"

#include <lmdb.h>

#include "LMDBwrapper/LMDB_handler.h"

#include <ctime>
#include <cmath>
#include <chrono>

#include "img_proc.h"
#include "boost_program_options.h"

#include <dlib/dnn.h>

#include <math.h>

using namespace std;
using namespace dlib;
using namespace cv;

using bpo = boost_program_options;

// TODO: проверить функцию проверки (на проценты)

// --- Здесь записаны основные сети для обучения ---

/* //Сеть 1 (testBacgrounds)
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            relu<con<25,3,3,2,2, // 22x22
                            max_pool<2,2,2,2,relu<con<20,5,5,2,2, // 90x90 -> 45x45
                            input<matrix<uchar>> // 183x183
                            >>>>>>>>;*/
// Сеть 2 (testBcgOneMoreLayer)
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<con<25,3,3,2,2, // 22x22 -> 11x11
                            max_pool<2,2,2,2,relu<con<20,5,5,2,2, // 90x90 -> 45x45
                            input<matrix<uchar>> // 183x183
                            >>>>>>>>;*/
// Сеть 3 (testBcgLessStrideBN | testBcg_LS_BN_BR | testBcg_LS_BN_BR_RC)
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<30,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<25,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<20,5,5,2,2,
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>;*/
// Сеть 4 (4 layers)
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<35,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<30,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<25,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<20,5,5,2,2,
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>;*/
// Сеть 5 (ядра 3х3)
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<30,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<25,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<20,3,3,2,2,
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>;*/
// сеть 6
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<50,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<45,2,2,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<35,3,3,1,1,
                            max_pool<2,2,2,2,relu<bn_con<con<30,5,5,2,2,
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>;*/
// сеть 7
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,relu<bn_con<con<50,2,2,1,1,   // 1x1
                            max_pool<2,2,2,2,relu<bn_con<con<45,2,2,1,1,   // 1x1
                            max_pool<2,2,2,2,relu<bn_con<con<35,3,3,1,1,   // 1x1 -> 1x1
                            max_pool<2,2,2,2,relu<bn_con<con<30,5,5,2,2,   // 6x6 -> 3x3
                            max_pool<2,2,2,2,relu<bn_con<con<20,15,15,3,3, // 29x29 -> 15x15
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
                            max_pool<2,2,2,2,relu<bn_con<con<256,3,3,1,1,  // 9x9 -> 5x5
                            max_pool<2,2,2,2,relu<bn_con<con<128,3,3,1,1,  // 22x22 -> 11x11
                            max_pool<2,2,2,2,relu<bn_con<con<64,3,3,1,1,   // 47x47 -> 24x24
                            max_pool<2,2,2,2,relu<bn_con<con<32,3,3,1,1,   // 97x97 -> 49x49
                            max_pool<2,2,2,2,relu<bn_con<con<16,3,3,1,1,   // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 10
/*using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<256,3,3,1,1,  // 9x9 -> 5x5
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,  // 22x22 -> 11x11
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,   // 47x47 -> 24x24
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,   // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,   // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 11
/*
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,  // 9x9 -> 5x5
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,   // 22x22 -> 11x11
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,   // 47x47 -> 24x24
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,   // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<8,3,3,1,1,    // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
// сеть 12
using net_type_2 = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<128,3,3,1,1,  // 8x8 -> 4x4
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,   // 19x19 -> 10x10
                            prelu<bn_con<con<64,3,3,1,1,                    // 21x21
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,   // 45x45 -> 23x23
                            prelu<bn_con<con<16,3,3,1,1,                    // 47x47
                            max_pool<2,2,2,2,prelu<bn_con<con<16,3,3,1,1,   // 97x97 -> 49x49
                            max_pool<2,2,2,2,prelu<bn_con<con<8,3,3,1,1,    // 198x198 -> 99x99
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>>>>>>>;
/*
// сеть 13
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 4x4 -> 2x2
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 12x12 -> 6x6
                            max_pool<2,2,2,2,prelu<bn_con<con<64,5,5,1,1,       // 27x27 -> 14x14
                            max_pool<2,2,2,2,prelu<bn_con<con<32,20,20,1,1,     // 62x62 -> 31x31
                            max_pool<2,2,2,2,prelu<bn_con<con<16,40,40,1,1,     // 161x161 -> 81x81
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>>;*/
/*// сеть 14
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 4x4 -> 2x2
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 12x12 -> 6x6
                            max_pool<2,2,2,2,prelu<bn_con<con<16,5,5,1,1,       // 27x27 -> 14x14
                            max_pool<2,2,2,2,prelu<bn_con<con<16,20,20,1,1,     // 62x62 -> 31x31
                            prelu<bn_con<con<8,40,40,2,2,                       // 81x81
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>;*/
// сеть 15
using net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<con<64,3,3,1,1,       // 7x7 -> 4x4
                            max_pool<2,2,2,2,prelu<bn_con<con<32,3,3,1,1,       // 17x17 -> 9x9
                            max_pool<2,2,2,2,prelu<bn_con<con<16,5,5,1,1,       // 37x37 -> 19x19
                            max_pool<2,2,2,2,prelu<bn_con<con<16,10,10,1,1,     // 82x82 -> 41x41
                            prelu<bn_con<con<8,20,20,2,2,                       // 91x91
                            input<matrix<uchar>>
                            >>>>>>>>>>>>>>>>>>>>>;

// --- Здесь записаны сети для обучения на частях исходного изображения (черновая сеть, определяющая начальные точки, а также сети для бровей, глаз и носа)
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
// черновая сеть 1
/*
using draft_net_type = loss_mean_squared_multioutput<
                            fc<62,
                            max_pool<2,2,2,2,prelu<bn_con<25,3,3,2,2, //  ->
                            max_pool<2,2,2,2,prelu<bn_con<20,5,5,2,2, //  ->
                            input<matrix<uchar>> // 200x200
                            >>>>>>>>;*/

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


// функция создания минипакетов
void create_minibatch (std::vector<matrix<uchar>> &mini_batch_samples, std::vector<dlib::matrix<float>> &mini_batch_labels, LMDBCursor* &cursor)
{
    // переменные строки со значениями, соотв. самого изображения и точек
    cv::Mat img;
    std::string matWithVecLabelAsStr;
    // матрица из учаров
    matrix<uchar> imgGray;
    std::vector<std::vector<short>> points;

    // цикл, заполняющий вектора с изображениями и точек
    for (int i = 0; i < bpo::mini_batch_size; ++i)
    {
        // помещаем курсор на значение ключа и получаем значения
        cursor->Next();

        // если курсор попал на конец обучающей выборки, то перемещаем его в начало
        if (std::stoi(cursor->key()) == bpo::images) cursor->SeekToFirst();

        // получаем строку из базы
        matWithVecLabelAsStr = cursor->value();

        // преобразуем строку в изображение и набор точек
        LMDB_handler::deserializeStrTo8UMatAndVecLabel(matWithVecLabelAsStr, img, points);

        // матрица из флоатов для сети
        dlib::matrix<float> train;
        train.set_size(points.size()*2,1); //  строки и колонки

        // заносим точки в вектор
        for (int y = 0; y < points.size(); ++y)
        {
            train(y * 2) = (float)(points[y][0]);
            train(y * 2 + 1) = (float)(points[y][1]);
        }

        points.clear();
        // трансформируем изображение
        int incor = img_proc::transform_image(img, train);

        if (!incor)
        {
            // переводим матрицу cv в изображение dlib
            assign_image(imgGray, cv_image<uchar>(img));
            // запушим в вектора для обучения
            mini_batch_samples.push_back(imgGray);
            mini_batch_labels.push_back(train);
        }
        else // если возникла ошибка с транформированием изображения, то игнорируем его результат и возвращаемся на шаг назад
        {
            i--;
        }
    }
}

// функция создания минипакетов для части изображения
template <class DT>
void create_minibatch (std::vector<matrix<uchar>> &mini_batch_samples, std::vector<dlib::matrix<float>> &mini_batch_labels, LMDBCursor* &cursor, DT &DraftNet)
{
    // переменные строки со значениями, соотв. самого изображения и точек
    cv::Mat img;
    std::string matWithVecLabelAsStr;
    // матрица из учаров
    matrix<uchar> imgGray;
    matrix<uchar> draftimgGray;
    std::vector<std::vector<short>> points;

    // цикл, заполняющий вектора с изображениями и точек
    for (int i = 0; i < bpo::mini_batch_size; ++i)
    {
        // помещаем курсор на значение ключа и получаем значения
        cursor->Next();

        // если курсор попал на конец обучающей выборки, то перемещаем его в начало
        if (std::stoi(cursor->key()) == bpo::images) cursor->SeekToFirst();

        // получаем строку из базы
        matWithVecLabelAsStr = cursor->value();

        // преобразуем строку в изображение и набор точек
        LMDB_handler::deserializeStrTo8UMatAndVecLabel(matWithVecLabelAsStr, img, points);

        // матрица из флоатов для черновой сети
        dlib::matrix<float> draft_train;
        draft_train.set_size(points.size()*2,1); //  строки и колонки

        // матрица из флоатов для обучаемой сети
        dlib::matrix<float> dlib_train;
        dlib_train.set_size(points.size()*2,1); //  строки и колонки

        // заносим точки в вектор
        for (int y = 0; y < points.size(); ++y)
        {
            dlib_train(y * 2) = (float)(points[y][0]);
            dlib_train(y * 2 + 1) = (float)(points[y][1]);
        }

        points.clear();

        // изменяем изображение по заданым параметрам
        int incor = img_proc::transform_image(img, dlib_train);

        if (!incor)
        {
            // переводим матрицу cv в изображение dlib
            assign_image(draftimgGray, cv_image<uchar>(img));

            draft_train = DraftNet(draftimgGray);

            cv::Mat image_patch;
            dlib::matrix<float> coords, draft_train_patch;
            coords.set_size(4,1);
            // задаём размер точек с патчем
            draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

            switch (bpo::cascade_num)
            {
                case 1: // левая бровь
                {
                    img_proc::get_left_eyebow_patch(img, image_patch, coords, draft_train, draft_train_patch, bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                    break;
                }
                case 2: // правая бровь
                {
                    img_proc::get_right_eyebow_patch(img, image_patch, coords, draft_train, draft_train_patch, bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                    break;
                }
                case 3: // правая бровь
                {
                    img_proc::get_nose_patch(img, image_patch, coords, draft_train, draft_train_patch, bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                    break;
                }
                case 4: // правая бровь
                {
                    img_proc::get_left_eye_patch(img, image_patch, coords, draft_train, draft_train_patch, bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                    break;
                }
                case 5: // правая бровь
                {
                    img_proc::get_right_eye_patch(img, image_patch, coords, draft_train, draft_train_patch, bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                    break;
                }
            }
            // преобразуем полученное изображение в cv
            assign_image(imgGray, cv_image<uchar>(image_patch));

            // запушим в вектора для обучения
            mini_batch_samples.push_back(imgGray);
            mini_batch_labels.push_back(draft_train_patch);
        }
        else // если возникла ошибка с транформированием изображения, то игнорируем его результат и возвращаемся на шаг назад
        {
            i--;
        }
    }
}

// функция обучения сети (вызывает создание минипакета и проводит один шаг для тренера, пока скорость обучения выше заданой)
template <class T, class TT>
void extract_and_process (T &trainer, TT &net)
{
    // настройка тренера
    trainer.be_verbose();
    trainer.set_iterations_without_progress_threshold(bpo::iteration_without_progress);
    trainer.set_learning_rate_shrink_factor(bpo::learning_rate_shrink);
    trainer.set_learning_rate(bpo::learning_rate);

    // задание файла синхронизации
    string synchName = bpo::snapshot_directory_path + "/" + bpo::network_name + "/" + "net_sync_" + bpo::network_name;
    trainer.set_synchronization_file(synchName, std::chrono::seconds(5000));
    // задание папки сохранения состояний сети
    string netSnapshotPath = bpo::snapshot_directory_path + "/" + bpo::network_name;
    create_directory(netSnapshotPath);

    // открываем базу для чтения
    LMDB Data;
    Data.Open(bpo::lmdb_folder_path, READ);

    cout << "Database opened." << endl;

    // создаём курсор базы данных
    LMDBCursor* cursor = Data.NewCursor();
    cursor->SeekToLast();

    std::string imgLast = cursor->key();

    cout << "Images in database: " << std::stoi(imgLast) << ". In training set: " << bpo::images << endl;

    // перемещаем курсор на первое изображение в базе
    int iter = 1;
    cursor->SeekToFirst();

    // для создания мини-пакетов создадим необходимые переменные
    std::vector<matrix<uchar>> mini_batch_samples;
    std::vector<dlib::matrix<float>> mini_batch_labels;
    mini_batch_samples.clear();
    mini_batch_labels.clear();

    // для обучения малой сети, подключим сеть, которая будет размечать первичные точки на изображении
    draft_net_type DraftNet;
    if (bpo::cascade_num != 0)
    {
        // открываем черновую сеть
        // путь к нужному файлу
        string DraftNetName = bpo::snapshot_directory_path + "/" + bpo::draft_network_name + "/" + bpo::draft_network_name + "_iter_" + bpo::draft_iter_name + ".dat";
        // загружаем полученную сеть
        deserialize(DraftNetName) >> DraftNet;
    }

    // пока скорость обучения больше заданого минимума
    while(trainer.get_learning_rate() >= bpo::required_learning_rate)
    {
        if(iter % 10000 == 0)
        {
            std::string snapshotFile = bpo::snapshot_directory_path + "/" + bpo::network_name + "/" + bpo::network_name + "_iter_" + std::to_string(iter) + ".dat";
            serialize(snapshotFile) << trainer.get_net();
        }

        // создаём один минипакет (в зависимости какую сеть обучаем, берём либо полное изображение, либо его часть)
        if (bpo::cascade_num != 0) create_minibatch(mini_batch_samples, mini_batch_labels, cursor, DraftNet);
        else create_minibatch(mini_batch_samples, mini_batch_labels, cursor);
        // тренируем один шаг
        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
        mini_batch_labels.clear();
        mini_batch_samples.clear();
        iter++;
    }
    // сохраняем результат работы
    net = trainer.get_net();
    net.clean();
    std::string finalNetName = bpo::snapshot_directory_path + "/" + bpo::network_name + "/" + bpo::network_name + "_iter_" + std::to_string(iter) + ".dat";
    serialize(finalNetName) << net;

    Data.Close();
    cout << "Training complete in: " << iter << " iterations."<< endl;

}

// проверка работоспособности сетей (использующих полное изображение или его части)
template <class T>
void test_network(T &net)
{
    // загрузка сети и подготовка изображений из тестового набора

    // строка с путём к файлу тестируемой сети
    string NetName = bpo::snapshot_directory_path + "/" + bpo::network_name + "/" + bpo::network_name + "_iter_" + bpo::iter_name + ".dat";
    // получаем тестируемую сеть
    deserialize(NetName) >> net;

    // подготавливаем данные для теста
    // открываем базу данных
    LMDB Data;
    Data.Open(bpo::lmdb_folder_path, READ);

    cout << "Database opened." << endl;

    // создаём курсор базы данных
    LMDBCursor* cursor = Data.NewCursor();

    // узнаём конечное число изображений
    cursor->SeekToLast();
    std::string imgLast = cursor->key();

    cout << "Images in database: " << std::stoi(imgLast) << ". In testing set: " << std::stoi(imgLast) - bpo::images << endl;

    // создаём представления ключа в бд с номером начала тестирующей выборки
    string tmpstr = std::to_string(bpo::images);
    int NumLength = 10 - tmpstr.length();
    for (int s = 0; s < NumLength; ++s)
    {
        tmpstr.insert(0,"0");
    }
    // помещаем курсор на значение ключа и получаем значения
    cursor->GetByKey(tmpstr);

    // создаём необходимые переменные
    // переменные для десериализации из базы
    std::string matWithVecLabelAsStr;
    cv::Mat img;
    std::vector<std::vector<short>> points;
    // для перевода в форму для сети
    matrix<uchar> imgGray;
    dlib::matrix<float> dlib_train, draft_train; // набор верхных точек от длиба и набор точек, полученных черновой сетью
    std::vector<matrix<uchar>> test_samples, test_samples_full; // изображения (если в тестах участвуют части изображений, то сохраняем полные изображения)
    std::vector<dlib::matrix<float>> test_labels; // готовый набор разметки, из делиба
    std::vector<dlib::matrix<float>> predicted_labels; // готовый набор разметки, полученный сетью
    test_samples.clear();
    test_labels.clear();
    test_samples_full.clear();
    test_labels.clear();
    predicted_labels.clear();
    // патчи изображения, левая бровь, правая, глаза и нос
    cv::Mat image_patch;
    // матрица с иходными координатами при вырезке патча (для его возвращения обратно)
    dlib::matrix<float> coords, all_coords, draft_train_patch, dlib_train_patch;
    coords.set_size(4,1);
    int temporary = std::stoi(imgLast) - bpo::images;
    all_coords.set_size(4,temporary);

    // точки для центров глаз
    dlib::point po, lcp, rcp, lcpd, rcpd;
    std::vector<dlib::point> eyes_centre;

    qsrand(QDateTime::currentMSecsSinceEpoch());

    draft_net_type DraftNet;
    if (bpo::cascade_num != 0)
    {
        // открываем черновую сеть
        // путь к нужному файлу
        string DraftNetName = bpo::snapshot_directory_path + "/" + bpo::draft_network_name + "/" + bpo::draft_network_name + "_iter_" + bpo::draft_iter_name + ".dat";
        // загружаем полученную сеть
        deserialize(DraftNetName) >> DraftNet;
    }

    cout << "Preparing dataset..." << endl;
    int counter = 0;
    while (cursor->key() != imgLast)
    {
        // получаем строку из базы
        matWithVecLabelAsStr = cursor->value();

        // преобразуем строку в изображение и набор точек
        LMDB_handler::deserializeStrTo8UMatAndVecLabel(matWithVecLabelAsStr, img, points);

        // переводим в матрицу из учаров
        assign_image(imgGray, cv_image<uchar>(img));
        test_samples_full.push_back(imgGray);

        // матрица длиба с точками
        dlib_train.set_size(points.size()*2,1); //  строки и колонки
        for (int y = 0; y < points.size(); ++y)
        {
            dlib_train(y * 2) = (float)(points[y][0]);
            dlib_train(y * 2 + 1) = (float)(points[y][1]);
        }

        switch (bpo::cascade_num)
        {
            case 0: // без каскада
            {
                test_labels.push_back(dlib_train);
                break;
            }
            case 1: // левая бровь
            {
                dlib_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);
                draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

                draft_train = DraftNet(imgGray);
                // получаем патч, координаты точек для обучения, а также исходные координаты обрамляющего прямоугольника
                img_proc::get_left_eyebow_patch(img, image_patch, coords, draft_train, draft_train_patch,  bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                all_coords(0,counter) = coords(0); all_coords(1,counter) = coords(1); all_coords(2,counter) = coords(2); all_coords(3,counter) = coords(3);
                // переводим матрицу cv в изображение dlib
                assign_image(imgGray, cv_image<uchar>(image_patch));
                test_labels.push_back(draft_train_patch);
                break;
            }
            case 2: // правая бровь
            {
                dlib_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);
                draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

                draft_train = DraftNet(imgGray);
                // получаем патч, координаты точек для обучения, а также исходные координаты обрамляющего прямоугольника
                img_proc::get_right_eyebow_patch(img, image_patch, coords, draft_train, draft_train_patch,  bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                all_coords(0,counter) = coords(0); all_coords(1,counter) = coords(1); all_coords(2,counter) = coords(2); all_coords(3,counter) = coords(3);
                // переводим матрицу cv в изображение dlib
                assign_image(imgGray, cv_image<uchar>(image_patch));
                test_labels.push_back(draft_train_patch);
                break;
            }
            case 3: // нос
            {
                dlib_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);
                draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

                draft_train = DraftNet(imgGray);
                // получаем патч, координаты точек для обучения, а также исходные координаты обрамляющего прямоугольника
                img_proc::get_nose_patch(img, image_patch, coords, draft_train, draft_train_patch,  bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                all_coords(0,counter) = coords(0); all_coords(1,counter) = coords(1); all_coords(2,counter) = coords(2); all_coords(3,counter) = coords(3);
                // переводим матрицу cv в изображение dlib
                assign_image(imgGray, cv_image<uchar>(image_patch));
                test_labels.push_back(draft_train_patch);
                break;
            }
            case 4: // левый глаз
            {
                dlib_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);
                draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

                draft_train = DraftNet(imgGray);
                // получаем патч, координаты точек для обучения, а также исходные координаты обрамляющего прямоугольника
                img_proc::get_left_eye_patch(img, image_patch, coords, draft_train, draft_train_patch,  bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                all_coords(0,counter) = coords(0); all_coords(1,counter) = coords(1); all_coords(2,counter) = coords(2); all_coords(3,counter) = coords(3);
                // переводим матрицу cv в изображение dlib
                assign_image(imgGray, cv_image<uchar>(image_patch));
                test_labels.push_back(draft_train_patch);
                break;
            }
            case 5: // правый глаз
            {
                dlib_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);
                draft_train_patch.set_size(2 * bpo::points_sizes(0, bpo::cascade_num - 1), 1);

                draft_train = DraftNet(imgGray);
                // получаем патч, координаты точек для обучения, а также исходные координаты обрамляющего прямоугольника
                img_proc::get_right_eye_patch(img, image_patch, coords, draft_train, draft_train_patch,  bpo::patches_sizes(0, bpo::cascade_num - 1), bpo::patches_sizes(1, bpo::cascade_num - 1), 0);
                all_coords(0,counter) = coords(0); all_coords(1,counter) = coords(1); all_coords(2,counter) = coords(2); all_coords(3,counter) = coords(3);
                // переводим матрицу cv в изображение dlib
                assign_image(imgGray, cv_image<uchar>(image_patch));
                test_labels.push_back(draft_train_patch);
                break;
            }
        }

        test_samples.push_back(imgGray);

        counter++;
        cursor->Next();
    }
    cout << "Dataset preparation complete." << endl;

    // пока не дошли до конца тестирующей выборки получаем изображения и точки

    cout << "Processing images..." << endl;

    // засекаем время выполнения
    boost::posix_time::ptime testTimemcs = boost::posix_time::microsec_clock::universal_time();

    for (int t = 0; t < test_samples.size(); ++t)
    {
        boost::posix_time::ptime testIterTimemcs = boost::posix_time::microsec_clock::universal_time();
        predicted_labels.push_back(net(test_samples[t]));
        if (!(t % 500)) cout << t << ": " << (boost::posix_time::microsec_clock::universal_time() - testIterTimemcs).total_microseconds() << " mcs." << endl;
    }

    cout << "Done! " << test_samples.size() << " images processed in: " << (boost::posix_time::microsec_clock::universal_time() - testTimemcs).total_microseconds() << " mcs." << endl;


    // выбор автоматической или ручной проверки

    char answer;
    cout << "Type 'M' to manual lookthrough and 'A' to auto" << endl;
    cin >> answer;

    dlib::matrix<float> true_labels_imgproc, net_labels_imgproc;

    if (answer == 'M') // просмотр полученных изображений вручную
    {
        //MEMO
        int difference_x = 0, difference_y = 0;
        // проверяем каждое изображение
        for (int i = 0; i < test_samples.size(); ++i)
        {
            difference_x = 0; difference_y = 0;
            true_labels_imgproc.set_size(predicted_labels[i].size(),1);
            net_labels_imgproc.set_size(predicted_labels[i].size(),1);
            if (bpo::cascade_num != 0) // если проверяем часть изображения, то восстанавливаем координаты на основном изображении
            {
                for (int j = 0; j < predicted_labels[i].size()/2; ++j)
                {
                    // расстановка правильных точек на полном изображении (синие)
                    true_labels_imgproc(j*2) = test_labels[i](j*2) * ((float)all_coords(2,i) - (float)all_coords(0,i)) / bpo::patches_sizes(0, bpo::cascade_num - 1) + all_coords(0,i);
                    true_labels_imgproc(j*2+1) = test_labels[i](j*2+1) * ((float)all_coords(3,i) - (float)all_coords(1,i)) / bpo::patches_sizes(1, bpo::cascade_num - 1) + all_coords(1,i);
                    // расстановка угаданых точек на полном изображении (красные)
                    net_labels_imgproc(j*2) = predicted_labels[i](j*2) * ((float)all_coords(2,i) - (float)all_coords(0,i)) / bpo::patches_sizes(0, bpo::cascade_num - 1) + all_coords(0,i);
                    net_labels_imgproc(j*2+1) = predicted_labels[i](j*2+1) * ((float)all_coords(3,i) - (float)all_coords(1,i)) / bpo::patches_sizes(1, bpo::cascade_num - 1) + all_coords(1,i);

                    // test
                    /*
                    difference_x = round(abs(test_labels[i](j*2) - predicted_labels[i](j*2)));
                    difference_y = round(abs(test_labels[i](j*2+1) - predicted_labels[i](j*2+1)));

                    cout << test_labels[i](j*2) << " - " << predicted_labels[i](j*2) << " = " << difference_x << endl;
                    cout << test_labels[i](j*2+1) << " - " << predicted_labels[i](j*2+1) << " = " << difference_y << endl << endl;*/

                }
                img_proc::display_image(test_samples_full[i], true_labels_imgproc, net_labels_imgproc);
            }
            else
            {
               img_proc::display_image(test_samples[i], test_labels[i], predicted_labels[i]);

               /*for (int j = 0; j < predicted_labels[i].size()/2; ++j)
               {
                   difference_x = round(abs(test_labels[i](j*2) - predicted_labels[i](j*2)));
                   difference_y = round(abs(test_labels[i](j*2+1) - predicted_labels[i](j*2+1)));

                   cout << test_labels[i](j*2) << " - " << predicted_labels[i](j*2) << " = " << difference_x << endl;
                   cout << test_labels[i](j*2+1) << " - " << predicted_labels[i](j*2+1) << " = " << difference_y << endl << endl;
               }*/

            }
            cout << "Type 'C' to continue and 'A' to auto" << endl;
            cin >> answer;
            if (answer == 'A') break;
        }

    }

    if (answer == 'A') // проверка всего тестового набора в автоматическом режиме
    {
        // разница в координатах точек
        int difference_x, difference_y;
        int barycenter[8];
        int corr_left_barycenter = 0, corr_right_barycenter = 0;
        int wrong_left_barycenter = 0, wrong_right_barycenter = 0;
        int num_right_3x3 = 0, num_right_5x5 = 0, num_right_10x10 = 0, num_right_euclid_3 = 0, num_right_euclid_5 = 0;
        int num_wrong_3x3 = 0, num_wrong_5x5 = 0, num_wrong_10x10 = 0, num_wrong_euclid_3 = 0, num_wrong_euclid_5 = 0;
        for (int i = 0; i < test_samples.size(); ++i)
        {
            for (int k = 4; k < 8; ++k) barycenter[k] = 0;

            for (int j = 0; j < test_labels[i].size()/2; ++j)
            {
                // вычисляем разницу
                difference_x = round(abs(test_labels[i](j*2) - predicted_labels[i](j*2)));
                difference_y = round(abs(test_labels[i](j*2+1) - predicted_labels[i](j*2+1)));

                /*cout << test_labels[i](j*2) << " - " << predicted_labels[i](j*2) << " = " << difference_x << endl;
                cout << test_labels[i](j*2+1) << " - " << predicted_labels[i](j*2+1) << " = " << difference_y << endl;
                cin.get();*/

                if (difference_x < 3 && difference_y < 3) num_right_3x3++;
                else num_wrong_3x3++;
                // для области 10х10
                if (difference_x < 5 && difference_y < 5) num_right_5x5++;
                else num_wrong_5x5++;
                // для области 3х3
                if (difference_x < 10 && difference_y < 10) num_right_10x10++;
                else num_wrong_10x10++;
                // евклидово расстояние 3
                if (sqrt(difference_x*difference_x + difference_y*difference_y) < 3) num_right_euclid_3++;
                else num_wrong_euclid_3++;
                // евклидово расстояние 5
                if (sqrt(difference_x*difference_x + difference_y*difference_y) < 5) num_right_euclid_5++;
                else num_wrong_euclid_5++;
                // вычисление центроида левого глаза
                if ((bpo::cascade_num == 4 || (bpo::cascade_num == 0) && j >= 19 && j <= 24))
                {
                    barycenter[0] += predicted_labels[i](j*2);
                    barycenter[1] += predicted_labels[i](j*2+1);
                    barycenter[2] += test_labels[i](j*2);
                    barycenter[3] += test_labels[i](j*2+1);
                }
                if ((bpo::cascade_num == 5 || (bpo::cascade_num == 0) && j >= 25 && j <= 30))
                {
                    barycenter[4] += predicted_labels[i](j*2);
                    barycenter[5] += predicted_labels[i](j*2+1);
                    barycenter[6] += test_labels[i](j*2);
                    barycenter[7] += test_labels[i](j*2+1);
                }
                for (int k = 0; k < 8; ++k) barycenter[k] /= 6;
                if (bpo::cascade_num == 4 || bpo::cascade_num == 0)
                {
                    if (sqrt(pow(abs(barycenter[0] - barycenter[2]),2) + pow(abs(barycenter[1] - barycenter[3]),2)) < 3) corr_left_barycenter++;
                    else wrong_left_barycenter++;
                }
                if (bpo::cascade_num == 5 || bpo::cascade_num == 0)
                {
                    if (sqrt(pow(abs(barycenter[4] - barycenter[6]),2) + pow(abs(barycenter[5] - barycenter[7]),2)) < 3) corr_right_barycenter++;
                    else wrong_right_barycenter++;
                }

            }
        }
        cout << "Wrong points in 3x3 square: " << num_wrong_3x3 << ", right points: " << num_right_3x3 << ". Accuracy: " << fixed << num_right_3x3/(double)(num_right_3x3+num_wrong_3x3) << endl;
        cout << "Wrong points in 5x5 square: " << num_wrong_5x5 << ", right points: " << num_right_5x5 << ". Accuracy: " << fixed << num_right_5x5/(double)(num_right_5x5+num_wrong_5x5) << endl;
        cout << "Wrong points in 10x10 square: " << num_wrong_10x10 << ", right points: " << num_right_10x10 << ". Accuracy: " << fixed << num_right_10x10/(double)(num_right_10x10+num_wrong_10x10) << endl;
        cout << "Wrong points in euclid dist 3: " << num_wrong_euclid_3 << ", right points: " << num_right_euclid_3 << ". Accuracy: " << fixed << num_right_euclid_3/(double)(num_right_euclid_3+num_wrong_euclid_3) << endl;
        cout << "Wrong points in euclid dist 5: " << num_wrong_euclid_5 << ", right points: " << num_right_euclid_5 << ". Accuracy: " << fixed << num_right_euclid_5/(double)(num_right_euclid_5+num_wrong_euclid_5) << endl;
        if (bpo::cascade_num == 4 || bpo::cascade_num == 0) cout << "Wrong left e_cent in euc dist 3: " << wrong_left_barycenter << ", right points: " << corr_left_barycenter << ". Accuracy: " << fixed << corr_left_barycenter/(double)(corr_left_barycenter+wrong_left_barycenter) << endl;
        if (bpo::cascade_num == 5 || bpo::cascade_num == 0) cout << "Wrong left e_cent in euc dist 3: " << wrong_right_barycenter << ", right points: " << corr_right_barycenter << ". Accuracy: " << fixed << corr_right_barycenter/(double)(corr_right_barycenter+wrong_right_barycenter) << endl;

    }

}

// основная функция для выбора параметров обучения, создания тренера для сетей и вызова дальнейших функций.
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    // вытаскиеваем параметры заданые при запуске
    boost_program_options::GetParams(argc,argv);

    // если не задан парметр для последовательного обучения нескольких сетей, то вызваем диалог с пользователем и ожидаем ввод ответа
    char answer = 'T';
    if (!bpo::multiply_network)
    {
        cout << "Enter 'F' to fill database, 'T' to train or 'V' to valueate results. Type 'C' to train networks for parts of images." << endl;
        cin >> answer;
    }
    if (answer == 'F') LMDB_handler::fillLMDB(); // заполнение базы
    if (answer == 'T') // обучение сети (выбор из нескольких сетей, для автоматического обучения)
    {      
        if (bpo::cascade_num > 0 && bpo::cascade_num < 3)
        {
            eye_bow_net net;
            dnn_trainer<eye_bow_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
        else
        if (bpo::cascade_num == 3)
        {
            nose_net net;
            dnn_trainer<nose_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
        else
        if (bpo::cascade_num > 3 && bpo::cascade_num < 6)
        {
            eye_net net;
            dnn_trainer<eye_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
        else
        switch (bpo::multiply_network_number)
        {
        case 0:
            {
                // объявление сети
                net_type net;
                // объявление тренера
                dnn_trainer<net_type,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));

                extract_and_process(trainer, net);
                break;
            }
        case 1:
            {
                // объявление сети
                net_type_2 net;
                // объявление тренера
                dnn_trainer<net_type_2,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));

                extract_and_process(trainer, net);
                break;
            }
        }

    }
    if (answer == 'V') // проверка сети
    {
        if (bpo::cascade_num > 0 && bpo::cascade_num < 3)
        {
            eye_bow_net net;
            test_network(net);
        }
        else
        if (bpo::cascade_num == 3)
        {
            nose_net net;
            test_network(net);
        }
        else
        if (bpo::cascade_num > 3 && bpo::cascade_num < 6)
        {
            eye_net net;
            test_network(net);
        }
        else
        switch (bpo::multiply_network_number)
        {
        case 0:
            {
                // объявление сети
                net_type net;
                test_network(net);
                break;
            }
        case 1:
            {
                // объявление сети
                net_type_2 net;
                test_network(net);
                break;
            }
        }
    }
    if (answer == 'E') // заполнение и обучение
    {
        LMDB_handler::fillLMDB();

        switch (bpo::multiply_network_number)
        {
        case 0:
            {
                // объявление сети
                net_type net;

                // объявление тренера
                dnn_trainer<net_type,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));

                extract_and_process(trainer, net);
                break;
            }
        case 1:
            {
                // объявление сети
                net_type_2 net;

                // объявление тренера
                dnn_trainer<net_type_2,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));

                extract_and_process(trainer, net);
                break;
            }
        }
    }
    if (answer == 'C') // обучение сетей для частей изображения (глаза, нос, брови)
    {
        // номер сети каскада 0 - без каскада, 1 - левая бровь, 2 - правая бровь, 3 - нос, 4 - левый глаз, 5 - правый глаз.
        if (bpo::cascade_num > 0 && bpo::cascade_num < 3)
        {
            eye_bow_net net;
            dnn_trainer<eye_bow_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
        if (bpo::cascade_num == 3)
        {
            nose_net net;
            dnn_trainer<nose_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
        if (bpo::cascade_num > 3 && bpo::cascade_num < 6)
        {
            eye_net net;
            dnn_trainer<eye_net,adam> trainer(net,adam(bpo::weight_decay, 0.9, 0.999));
            extract_and_process(trainer, net);
        }
    }
    std::terminate();
    return 0;
}

