#include "img_proc.h"

#include "boost_program_options.h"

using bpo = boost_program_options;

img_proc::img_proc() {}

// функция вывода изображения и набора точек на экран
void img_proc::display_image(dlib::matrix<uchar> imgGray, dlib::matrix<float> train)
{
    // окно и точка на нём
    dlib::image_window win;
    dlib::point po;
    // выводим изображение на экран
    win.clear_overlay();
    win.set_image(imgGray);
    // расставим точки на слой
    for (int j = 0; j < train.size()/2; ++j)
    {
        // расстановка точек (синие)
        po.x() = train(j*2);
        po.y() = train(j*2+1);
        win.add_overlay(dlib::image_window::overlay_circle(po, 2, dlib::rgb_pixel(0,0,255)));
        win.add_overlay(po, po, dlib::rgb_pixel(0,0,255));
        //cin.get();
        //cout << "Test " << j << ": x - " << po.x() << " y - " << po.y() << endl;
    }
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();
}

// функция вывода изображения и двух наборов точек на экран
void img_proc::display_image(dlib::matrix<uchar> imgGray, dlib::matrix<float> train, dlib::matrix<float> test)
{
    // окно и точка на нём
    dlib::image_window win;
    dlib::point po;
    // выводим изображение на экран
    win.clear_overlay();
    win.set_image(imgGray);
    // расставим точки на слой
    for (int j = 0; j < train.size()/2; ++j)
    {
        // расстановка точек (синие)
        po.x() = train(j*2);
        po.y() = train(j*2+1);
        win.add_overlay(dlib::image_window::overlay_circle(po, 2, dlib::rgb_pixel(0,0,255)));
        win.add_overlay(po, po, dlib::rgb_pixel(0,0,255));
        // расстановка точек (красные)
        po.x() = test(j*2);
        po.y() = test(j*2+1);
        win.add_overlay(dlib::image_window::overlay_circle(po, 2, dlib::rgb_pixel(255,0,0)));
        win.add_overlay(po, po, dlib::rgb_pixel(255,0,0));
        //cout << "Test " << j << ": x - " << po.x() << " y - " << po.y() << endl;
    }
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

}

// афинное преобразование треугольной области на изображении
void img_proc::triangle_affine(std::vector<cv::Point2f> &srcTri, std::vector<cv::Point2f> &dstTri, cv::Mat &img, cv::Mat &img2)
{
    // перевод в целочисленные
    for(int i = 0; i < 3; i++)
    {
        srcTri[i] = cv::Point2f((int)srcTri[i].x, (int)srcTri[i].y);
        dstTri[i] = cv::Point2f((int)dstTri[i].x, (int)dstTri[i].y);
    }
    // ограничивающие треугольники прямоугольники
    cv::Rect r1 = boundingRect(srcTri);
    cv::Rect r2 = boundingRect(dstTri);
    // копируем часть изображения на временную матрицу
    cv::Mat imgCrop;
    img(r1).copyTo(imgCrop);
    // вектора для координат в ограничивающем прямоугольнике
    std::vector<cv::Point2f> srcTriCrop, dstTriCrop;
    std::vector<cv::Point> dstTriCropInt;

    for(int i = 0; i < 3; i++)
    {
        srcTriCrop.push_back(cv::Point2f(srcTri[i].x - r1.x, srcTri[i].y -  r1.y));
        dstTriCrop.push_back(cv::Point2f(dstTri[i].x - r2.x, dstTri[i].y - r2.y));

        dstTriCropInt.push_back(cv::Point((int)(dstTri[i].x - r2.x), (int)(dstTri[i].y - r2.y)) );
    }

    // получим матрицу афинного преобразования
    cv::Mat warp_mat = getAffineTransform(srcTriCrop, dstTriCrop);
    // заполняем нулями и преобразуем
    cv::Mat imgWarped = cv::Mat::zeros(r2.height, r2.width, imgCrop.type());
    cv::warpAffine(imgCrop, imgWarped, warp_mat, imgWarped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    // создаём маску
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_8U);
    cv::fillConvexPoly(mask, dstTriCropInt, cv::Scalar(255, 255, 255), 8, 0);
    // копируем по маске
    imgWarped.copyTo(img2(r2), mask);

    warp_mat.release();
    imgWarped.release();
    imgCrop.release();
}

// получить наборы координат закрытых/полузакрытых/прикрытих глаз
void img_proc::get_eye_points(std::vector<cv::Point2f> &close, std::vector<cv::Point2f> &close2, dlib::matrix<float> &train, double coeff)
{
    // точки уголков глаз и верхнего века
    /*
     *   1 2 3 4 5    брови    8 9 10 11 12
     *    0 a b  6   углы глаз  7 a  b  13
     *  фиксируем две центральные точки верхнего века и проходим по треугольникам
     *  (01a 12a 23a) - 1й цикл, a3b - отдельно, (b34 b45 b56) - 2й цикл
     * */
    close.clear();
    close.push_back(cv::Point2f(train(38),train(39)));
    for (int i = 0; i < 9; i += 2)
    {
        close.push_back(cv::Point2f(train(i),train(i+1)));
    }
    close.push_back(cv::Point2f(train(44),train(45)));
    close.push_back(cv::Point2f(train(50),train(51)));
    for (int i = 10; i < 19; i += 2)
    {
        close.push_back(cv::Point2f(train(i),train(i+1)));
    }
    close.push_back(cv::Point2f(train(56),train(57)));
    // две точки верхнего века обоих глаз, плюс изменение координат в векторе лейблов
    close2.clear();
    close2.push_back(cv::Point2f(train(40), train(41))); // 0
    close2.push_back(cv::Point2f(train(42), train(43))); // 1
    close2.push_back(cv::Point2f(train(52), train(53))); // 2
    close2.push_back(cv::Point2f(train(54), train(55))); // 3
    close2.push_back(cv::Point2f(train(40), train(49) - (train(49)-train(41))*coeff)); // 4
    close2.push_back(cv::Point2f(train(42), train(47) - (train(47)-train(43))*coeff)); // 5
    close2.push_back(cv::Point2f(train(52), train(61) - (train(61)-train(53))*coeff)); // 6
    close2.push_back(cv::Point2f(train(54), train(59) - (train(59)-train(55))*coeff)); // 7
}

// изменить глаз
void img_proc::set_triangles_and_transform(cv::Mat &img, cv::Mat &img2, std::vector<cv::Point2f> &close, std::vector<cv::Point2f> &close2, int left_right)
{
    std::vector<cv::Point2f> srcTri;
    std::vector<cv::Point2f> dstTri;

    for (int i = 0; i < 3; i++)
    {
        srcTri.push_back(close[i + left_right*7]);
        srcTri.push_back(close[i + 1 + left_right*7]);
        srcTri.push_back(close2[0 + left_right*2]);

        dstTri.push_back(close[i + left_right*7]);
        dstTri.push_back(close[i + 1 + left_right*7]);
        dstTri.push_back(close2[4 + left_right*2]);

        triangle_affine(srcTri, dstTri, img, img2);
        srcTri.clear();
        dstTri.clear();
    }

    srcTri.push_back(close2[0 + left_right*2]);
    srcTri.push_back(close[3 + left_right*7]);
    srcTri.push_back(close2[1 + left_right*2]);

    dstTri.push_back(close2[4 + left_right*2]);
    dstTri.push_back(close[3 + left_right*7]);
    dstTri.push_back(close2[5 + + left_right*2]);

    triangle_affine(srcTri, dstTri, img, img2);
    srcTri.clear();
    dstTri.clear();

    for (int i = 3; i < 6; i++)
    {
        srcTri.push_back(close[i + left_right*7]);
        srcTri.push_back(close[i + 1 + left_right*7]);
        srcTri.push_back(close2[1 + left_right*2]);

        dstTri.push_back(close[i + left_right*7]);
        dstTri.push_back(close[i + 1 + left_right*7]);
        dstTri.push_back(close2[5 + left_right*2]);

        triangle_affine(srcTri, dstTri, img, img2);
        srcTri.clear();
        dstTri.clear();
    }
}

// функции помещения изображения на фон
void img_proc::put_on_background(cv::Mat &img, dlib::matrix<float> &train)
{
    // выбор фона от 1 до 4
    qsrand(QDateTime::currentMSecsSinceEpoch());
    int rnd1 = 1 + qrand() % 4; // выбор фона 1 - 4

    // выбор размера фона (должен быть больше исходного на от 1 до  10)
    int tmp = img.cols;
    if (tmp < img.rows) tmp = img.rows;

    qsrand(QDateTime::currentMSecsSinceEpoch()+1);
    int rnd2 = tmp + 1 + qrand() % 10; // выбор размера фона 201-224 + коэфф для растяжения исходного

    // загружаем фон с заданым размером
    cv::Mat bcg = cv::imread(boost_program_options::background_directory_path + "/" + std::to_string(rnd1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);

    // вычисляем разницу в координатах и перемещаем картинку из базы в фон
    cv::resize(bcg,bcg,cv::Size(rnd2,rnd2));

    // помещаем в центр
    //int diffx = ((rnd2-cols)/2);
    //int diffy = ((rnd2-rows)/2);

    // случайное размещение
    int diffx = 1 + qrand() % (rnd2-img.cols);
    int diffy = 1 + qrand() % (rnd2-img.rows);
    img.copyTo(bcg(cv::Rect(diffx, diffy, img.cols, img.rows)));
    bcg.copyTo(img);
    cv::resize(img,img,cv::Size(200,200));
    // соответственно меняем координаты точек
    for (int y = 0; y < train.size()/2; ++y)
    {
        // размещение и изменение размера до 200х200
        train(y * 2) = (float)(((train(y*2) + diffx) * 200)/rnd2);
        train(y * 2 + 1) = (float)(((train(y*2+1) + diffy) * 200)/rnd2);

    }
}

// изменить размер изображения
void img_proc::resize_image(cv::Mat &img, dlib::matrix<float> &train)
{
    // случайное значение растяжения
    qsrand(QDateTime::currentMSecsSinceEpoch());
    float rnd = 0.01 * (90 + qrand() % 11); // растяжение

    cv::resize(img,img,cv::Size(rnd*img.cols,rnd*img.rows));

    for (int y = 0; y < train.size()/2; ++y)
    {
        train(y * 2) = (float)(((train(y*2) * rnd)));
        train(y * 2 + 1) = (float)(((train(y*2+1) * rnd)));
    }
}

// отображение изображения
void img_proc::flip_image(cv::Mat &img, dlib::matrix<float> &train)
{
    float trainFlipXtmp;
    float trainFlipYtmp;

    // отображаем изображение
    cv::flip(img, img, 1);

    // отображаем брови
    for (int y = 0; y < 5; ++y)
    {
        trainFlipXtmp = train(y * 2);
        trainFlipYtmp = train(y * 2 + 1);

        train(y * 2) = (float)(img.cols - train(18 - y*2) - 1);
        train(y * 2 + 1) = train(19 - y * 2);

        train(18 - y * 2) = (float)(img.cols - trainFlipXtmp - 1);
        train(19 - y * 2) = trainFlipYtmp;
    }

    for (int y = 19; y < 23; ++y)
    {
        trainFlipXtmp = train(y * 2);
        trainFlipYtmp = train(y * 2 + 1);

        train(y * 2) = (float)(img.cols - train(94 - y*2) - 1);
        train(y * 2 + 1) = train(95 - y * 2);

        train(94 - y * 2) = (float)(img.cols - trainFlipXtmp - 1);
        train(95 - y * 2) = trainFlipYtmp;
    }

    for (int y = 23; y < 25; ++y)
    {
        trainFlipXtmp = train(y * 2);
        trainFlipYtmp = train(y * 2 + 1);

        train(y * 2) = (float)(img.cols - train(106 - y*2) - 1);
        train(y * 2 + 1) = train(107 - y * 2);

        train(106 - y * 2) = (float)(img.cols - trainFlipXtmp - 1);
        train(107 - y * 2) = trainFlipYtmp;
    }

    for (int y = 14; y < 16; ++y)
    {
        trainFlipXtmp = train(y * 2);
        trainFlipYtmp = train(y * 2 + 1);

        train(y * 2) = (float)(img.cols - train(64 - y*2) - 1);
        train(y * 2 + 1) = train(65 - y * 2);

        train(64 - y * 2) = (float)(img.cols - trainFlipXtmp - 1);
        train(65 - y * 2) = trainFlipYtmp;
    }

    for (int y = 10; y < 14; ++y)
    {
        train(y * 2) = (float)(img.cols - train(y*2) - 1);
    }

    train(32) = (float)(img.cols - train(32) - 1);
}

// поворот изображения
void img_proc::rotate(cv::Mat &img, dlib::matrix<float> &train)
{
    qsrand(QDateTime::currentMSecsSinceEpoch());
    int rnd = -30 + qrand() % 61; // угол поворота
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(img.cols/2, img.rows/2), rnd, 1);
    cv::warpAffine(img, img, M, cv::Size(img.cols,img.rows));

    for (int y = 0; y < train.size()/2; ++y)
    {
        float tmpx = train(y * 2);
        float tmpy = train(y * 2 + 1);
        train(y * 2) = (float)(M.at<double>(0,0)*tmpx + M.at<double>(0,1)*tmpy + M.at<double>(0,2));
        train(y * 2 + 1) = (float)(M.at<double>(1,0)*tmpx + M.at<double>(1,1)*tmpy + M.at<double>(1,2));
    }
}

// закрытие глаз
void img_proc::reshape_eyes(cv::Mat &img, dlib::matrix<float> &train)
{
    // точки глаз для афинного преобразования
    std::vector<cv::Point2f> close, close2;

    // случайные значения
    qsrand(QDateTime::currentMSecsSinceEpoch());
    int rnd = qrand() % 100; // мера закрытия глаза
    qsrand(QDateTime::currentMSecsSinceEpoch() + 1);
    int rnd2 = qrand() % bpo::eye_chance; // вероятность закрытия глаз

    // перевод матрицы в дополнительную, чтобы не испортить триангуляцией первую
    cv::Mat img2;
    img.copyTo(img2);

    double coeff = 1;

    if (rnd < 55) // полное закрытие
    {
        coeff = 0.1;
        get_eye_points(close, close2, train, coeff);
        //cout << "fully closed" << endl;
    }
    else
    if (rnd >= 80) // небольшое закрытие
    {
        coeff = 0.7;
        get_eye_points(close, close2, train, coeff);
        //cout << "nearly closed" << endl;
    }
    else // наполовину
    {
        coeff = 0.5;
        get_eye_points(close, close2, train, coeff);
        //cout << "half closed" << endl;
    }

    // выбираем какие глаза закрыть
    if (rnd2 < 75)
    {
        train(41) = train(49) - (train(49)-train(41))*coeff;
        train(43) = train(47) - (train(47)-train(43))*coeff;
        set_triangles_and_transform(img, img2, close, close2, 0);
    }
    if ((rnd2 < 50) || (rnd2 >= 75))
    {
        train(53) = train(61) - (train(61)-train(53))*coeff;
        train(55) = train(59) - (train(59)-train(55))*coeff;
        set_triangles_and_transform(img, img2, close, close2, 1);
    }

    img2.copyTo(img);
}

// функция для выбора патча левого глаза
void img_proc::get_left_eye_patch(cv::Mat &img, cv::Mat &leye_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord)
{
    // принимаем изображение, матрицу для патча и точки.
    // точки от 19 до 24 включительно образуют левый глаз. Выбрать минимальные и максимальные координаты, добавить N пикселов к краям и выделить область.
    // поменять координаты точек.
    // точки для нахождения максимума и минимума
    dlib::point minp, maxp;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < train_patch.size()/2; ++j)
    {
        if (train(j*2+38) < minp.x()) minp.x() = train(j*2+38);
        if (train(j*2+38) > maxp.x()) maxp.x() = train(j*2+38);
        if (train(j*2+39) < minp.y()) minp.y() = train(j*2+39);
        if (train(j*2+39) > maxp.y()) maxp.y() = train(j*2+39);
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
    coords(0) = minp.x(); coords(1) = minp.y(); coords(2) = maxp.x(); coords(3) = maxp.y();

    // меняем размер и координаты точек
    if (!getcoord)
    {
        leye_patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
        cv::resize(leye_patch, leye_patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);
    }
    for(int j = 0; j < train_patch.size()/2; ++j) // если необходимы только координаты
    {
        train_patch(j*2) = (train(j*2+38)-minp.x())*(patch_size_x/((float)maxp.x() - (float)minp.x()));
        train_patch(j*2+1) = (train(j*2+39)-minp.y())*(patch_size_y/((float)maxp.y() - (float)minp.y()));
    }
}
// функция для выбора патча правого глаза
void img_proc::get_right_eye_patch(cv::Mat &img, cv::Mat &reye_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord)
{
    // принимаем изображение, матрицу для патча и точки.
    // точки от 25 до 30 включительно образуют правый глаз. Выбрать минимальные и максимальные координаты, добавить N пикселов к краям и выделить область.
    // поменять координаты точек.
    // точки для нахождения максимума и минимума
    dlib::point minp, maxp;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < train_patch.size()/2; ++j)
    {
        if (train(j*2+50) < minp.x()) minp.x() = train(j*2+50);
        if (train(j*2+50) > maxp.x()) maxp.x() = train(j*2+50);
        if (train(j*2+51) < minp.y()) minp.y() = train(j*2+51);
        if (train(j*2+51) > maxp.y()) maxp.y() = train(j*2+51);
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
    coords(0) = minp.x(); coords(1) = minp.y(); coords(2) = maxp.x(); coords(3) = maxp.y();

    // меняем размер и координаты точек
    if (!getcoord)
    {
        reye_patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
        cv::resize(reye_patch, reye_patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);
    }
    for(int j = 0; j < train_patch.size()/2; ++j) // если необходимы только координаты
    {
        train_patch(j*2) = (train(j*2+50)-minp.x())*(patch_size_x/((float)maxp.x() - (float)minp.x()));
        train_patch(j*2+1) = (train(j*2+51)-minp.y())*(patch_size_y/((float)maxp.y() - (float)minp.y()));
    }
}
// функция для выбора патча носа
void img_proc::get_nose_patch(cv::Mat &img, cv::Mat &nose_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord)
{
    // принимаем изображение, матрицу для патча и точки.
    // точки от 10 до 18 включительно образуют нос. Выбрать минимальные и максимальные координаты, добавить N пикселов к краям и выделить область.
    // поменять координаты точек.
    // точки для нахождения максимума и минимума
    dlib::point minp, maxp;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < train_patch.size()/2; ++j)
    {
        if (train(j*2+20) < minp.x()) minp.x() = train(j*2+20);
        if (train(j*2+20) > maxp.x()) maxp.x() = train(j*2+20);
        if (train(j*2+21) < minp.y()) minp.y() = train(j*2+21);
        if (train(j*2+21) > maxp.y()) maxp.y() = train(j*2+21);
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
    coords(0) = minp.x(); coords(1) = minp.y(); coords(2) = maxp.x(); coords(3) = maxp.y();
    // меняем размер и координаты точек
    if (!getcoord)
    {
        nose_patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
        cv::resize(nose_patch, nose_patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);
    }
    for(int j = 0; j < train_patch.size()/2; ++j) // если необходимы только координаты
    {
        train_patch(j*2) = (train(j*2+20)-minp.x())*(patch_size_x/((float)maxp.x() - (float)minp.x()));
        train_patch(j*2+1) = (train(j*2+21)-minp.y())*(patch_size_y/((float)maxp.y() - (float)minp.y()));
    }
}
// функция для выбора патча левой брови
void img_proc::get_left_eyebow_patch(cv::Mat &img, cv::Mat &leyebow_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord)
{
    // принимаем изображение, матрицу для патча и точки.
    // точки от 0 до 4 включительно образуют левую бровь. Выбрать минимальные и максимальные координаты, добавить N пикселов к краям и выделить область.
    // поменять координаты точек.
    // точки для нахождения максимума и минимума
    dlib::point minp, maxp;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < train_patch.size()/2; ++j)
    {
        if (train(j*2) < minp.x()) minp.x() = train(j*2);
        if (train(j*2) > maxp.x()) maxp.x() = train(j*2);
        if (train(j*2+1) < minp.y()) minp.y() = train(j*2+1);
        if (train(j*2+1) > maxp.y()) maxp.y() = train(j*2+1);
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
    coords(0) = minp.x(); coords(1) = minp.y(); coords(2) = maxp.x(); coords(3) = maxp.y();

    // меняем размер и координаты точек
    if (!getcoord)
    {
        leyebow_patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
        cv::resize(leyebow_patch, leyebow_patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);
    }
    for(int j = 0; j < train_patch.size()/2; ++j) // если необходимы только координаты
    {
        train_patch(j*2) = (train(j*2)-minp.x())*(patch_size_x/((float)maxp.x() - (float)minp.x()));
        train_patch(j*2+1) = (train(j*2+1)-minp.y())*(patch_size_y/((float)maxp.y() - (float)minp.y()));
    }
}
// функция для выбора патча правой брови
void img_proc::get_right_eyebow_patch(cv::Mat &img, cv::Mat &reyebow_patch, dlib::matrix<float> &coords, dlib::matrix<float> &train, dlib::matrix<float> &train_patch, int patch_size_x, int patch_size_y, bool getcoord)
{
    // принимаем изображение, матрицу для патча и точки.
    // точки от 5 до 9 включительно образуют правую бровь. Выбрать минимальные и максимальные координаты, добавить N пикселов к краям и выделить область.
    // поменять координаты точек.
    // точки для нахождения максимума и минимума
    dlib::point minp, maxp;

    // задаём начальные значения точек максимума и минимума
    minp.x() = img.cols; minp.y() = img.rows; maxp.x() = 0; maxp.y() = 0;
    // находим координаты для ограничивающего прямоугольника
    for(int j = 0; j < train_patch.size()/2; ++j)
    {
        if (train(j*2+10) < minp.x()) minp.x() = train(j*2+10);
        if (train(j*2+10) > maxp.x()) maxp.x() = train(j*2+10);
        if (train(j*2+11) < minp.y()) minp.y() = train(j*2+11);
        if (train(j*2+11) > maxp.y()) maxp.y() = train(j*2+11);
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
    coords(0) = minp.x(); coords(1) = minp.y(); coords(2) = maxp.x(); coords(3) = maxp.y();
    // меняем размер и координаты точек
    if (!getcoord)
    {
        reyebow_patch = img(cv::Rect(minp.x(), minp.y(), maxp.x() - minp.x(), maxp.y() - minp.y()));
        cv::resize(reyebow_patch, reyebow_patch, cv::Size(patch_size_x,patch_size_y), cv::INTER_NEAREST);
    }
    for(int j = 0; j < train_patch.size()/2; ++j) // если необходимы только координаты
    {
        train_patch(j*2) = (train(j*2+10)-minp.x())*(patch_size_x/((float)maxp.x() - (float)minp.x()));
        train_patch(j*2+1) = (train(j*2+11)-minp.y())*(patch_size_y/((float)maxp.y() - (float)minp.y()));
    }
}

// функция применения всех изменений
// изменить изображение и ключевые точки
int img_proc::transform_image(cv::Mat &img, dlib::matrix<float> &train)
{
    dlib::matrix<uchar> imgGray;
    // создаём копии
    cv::Mat image;
    dlib::matrix<float> training;
    img.copyTo(image);
    training = train;

    if (bpo::random_cropper_enabled)
    {
        img_proc::rotate(image,training);
        //assign_image(imgGray, dlib::cv_image<uchar>(img));
        //img_proc::display_image(imgGray,train);
        if (borders_check(image, training)) return 1;
    }

    if (bpo::backgrounds)
    {
        img_proc::put_on_background(image,training);
        //assign_image(imgGray, dlib::cv_image<uchar>(img));
        //img_proc::display_image(imgGray,train);
        if (borders_check(image, training)) return 1;
    }

    if (bpo::random_cropper_flip)
    {
        img_proc::flip_image(image,training);
        //assign_image(imgGray, dlib::cv_image<uchar>(img));
        //img_proc::display_image(imgGray,train);
        if (borders_check(image, training)) return 1;
    }

    if (bpo::eye_reshaper)
    {
        img_proc::reshape_eyes(image,training);
        //assign_image(imgGray, dlib::cv_image<uchar>(img));
        //img_proc::display_image(imgGray,train);
        if (borders_check(image, training)) return 1;
    }

    if (bpo::random_cropper_resize)
    {
        img_proc::resize_image(image,training);
        //assign_image(imgGray, dlib::cv_image<uchar>(img));
        //img_proc::display_image(imgGray,train);
        if (borders_check(image, training)) return 1;
    }
    image.copyTo(img);
    train = training;
    if (bpo::check_image)
    {
        assign_image(imgGray, dlib::cv_image<uchar>(img));
        img_proc::display_image(imgGray, train);

    }
    return 0;
}

bool img_proc::borders_check(cv::Mat &img, dlib::matrix<float> &train)
{
    // проверяем, что точки внутренние
    bool out_of_borders = 0;
    for (int y = 0; y < train.size()/2; ++y)
    {
        if (train(y * 2) > img.cols) out_of_borders = 1;
        if (train(y * 2 + 1) > img.rows) out_of_borders = 1;
        if (train(y * 2) < 0) out_of_borders = 1;
        if (train(y * 2 + 1) < 0) out_of_borders = 1;
    }
    return out_of_borders;
}
