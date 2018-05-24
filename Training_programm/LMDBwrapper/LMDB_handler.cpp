#include "LMDB_handler.h"

using bpo = boost_program_options;

LMDB_handler::LMDB_handler(){}

LMDB* LMDB_handler::NewLMDB()
{
    return new LMDB();
}

// сериализия изображения с вектором точек в строку (изображение матрица cv, вектор векторов из шортов [координаты точек в шорте], преобразованая строка с вектором и изображением)
void LMDB_handler::serialize8UMatWithVecLabelToStr(const cv::Mat& img, const std::vector<std::vector<short>> &labelVector, std::string& matWithVectorAsStr)
{
    unsigned int channels = img.channels();             // каналы изображения
    unsigned int rows = img.rows;                       // колонки и строки
    unsigned int cols = img.cols;
    unsigned int imgSize = channels * rows * cols;      // размер изображения

    //cout << "channels: " << channels << endl << "rows: " << rows << endl << "cols: " << cols << endl << "img_size: " << imgSize << endl;

    // подготавливаем строчку
    matWithVectorAsStr.clear();
    matWithVectorAsStr.resize(imgSize);

    // набор битов для информации об изображении и точках вектора
    std::bitset<32> channelsBits(channels);
    std::bitset<32> rowsBits(rows);
    std::bitset<32> colsBits(cols);
    std::bitset<16> labelVectorBit;

    // заполняем строку изображением (изображение записывается построчно)
    for (int h = 0; h < rows; ++h)
    {
        const uchar* ptr = img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cols; ++w)
        {
            for (int c = 0; c < channels; ++c)
            {
                int index = (c * rows + h) * cols + w;
                matWithVectorAsStr[index] = static_cast<char>(ptr[img_index++]);

            }
        }
    }

    // заполняем данные о изображении
    matWithVectorAsStr.insert(0, channelsBits.to_string());
    matWithVectorAsStr.insert(0, rowsBits.to_string());
    matWithVectorAsStr.insert(0, colsBits.to_string());

    // заполняем набор битов данными о точках (y[30], x[30], y[29], x[29], .. , y[0], x[0]) обратный .dat файлу
    for (int k = 30; k >= 0; --k)
    {
        labelVectorBit = labelVector[k][1];
        //cout << k << ": y: " << labelVector[k][1]  << "  |  " << labelVectorBit << endl;
        matWithVectorAsStr.insert(0, labelVectorBit.to_string());
        labelVectorBit = labelVector[k][0];
        //cout << k << ": x: " << labelVector[k][0]  << "  |  " << labelVectorBit << endl;
        matWithVectorAsStr.insert(0, labelVectorBit.to_string());

    }
}

// десериализация полученных строк обратно в изображения
void LMDB_handler::deserializeStrTo8UMatAndVecLabel(const std::string& matWithVectorAsStr, cv::Mat& img,  std::vector<std::vector<short>>  &labelVector)
{
    std::string labelVectorStr = matWithVectorAsStr.substr(0, 31*32);   // подстрока со значениями
    std::string colsStr = matWithVectorAsStr.substr(31*32, 32);         // подстрока с данными о столбцах
    std::string rowsStr = matWithVectorAsStr.substr(32*32, 32);         // подстрока с данными о строках
    std::string channelsStr = matWithVectorAsStr.substr(32*33, 32);     // подстрока с данными о каналах

    // соответвтующие bitset'ы
    std::bitset<32> channelsBits(channelsStr);
    std::bitset<32> rowsBits(rowsStr);
    std::bitset<32> colsBits(colsStr);
    // соответствующие переменные
    int channels = (int)channelsBits.to_ulong();
    int rows = (int)rowsBits.to_ulong();
    int cols = (int)colsBits.to_ulong();

    std::vector<short> tmp = {0,0};

    labelVector.clear(); // очищаем от предыдущих значений

    // получаем вектор точек (x[0], y[0], x[1], y[1], .. , x[30], y[30]) аналогичный .dat файлу
    for (int k = 0; k < 31; ++k)
    {
        std::bitset<16> bits(labelVectorStr.substr(k*32, 16));
        tmp[0] = (short)bits.to_ulong();
        //cout << k << ": x: " << tmp[0] << "  |  " << bits << endl;
        std::bitset<16> bits1(labelVectorStr.substr(k*32 + 16, 16));
        tmp[1] = (short)bits1.to_ulong();
        //cout << k << ": y: " << tmp[1] << "  |  " << bits1 << endl;
        labelVector.push_back(tmp);
    }

    // число битов, отведённое под информацию о точках и изображении
    int metaDataSize = 34 * 32;
    // обработка каналов изображения и создание соответствующего изображения
    if(channels == 1)
        img.create(rows, cols, CV_8UC1);
    else if(channels == 3)
        img.create(rows, cols, CV_8UC3);

    // вектор с матрицами каналов
    std::vector<cv::Mat> imgChnls;

    // заполнение матриц изображения
    int num = 0;
    for(int k = 0; k < channels; ++k)
    {
        cv::Mat tmpImg(rows, cols, CV_8UC1);
        for(int r = 0; r < rows; ++r)
        {
            for(int c = 0; c < cols; ++c)
            {
                tmpImg.at<uchar>(r, c) = (int)static_cast<uchar>(matWithVectorAsStr[num + metaDataSize]);
                num++;
            }
        }
        imgChnls.push_back(tmpImg);
    }
    // подготовка финального изображения
    if(channels == 1)
        img = imgChnls[0];
    else if(channels == 3)
        cv::merge(imgChnls, img);
    /*cv::imshow("test",img);
    cout << channels << "|" << rows << "x" << cols << endl;
    cv::waitKey(0);*/
}

// заполнение базы (заполнение происходит случайным образом)
void LMDB_handler::fillLMDB()
{
    // Создаём новое хранилище
    LMDB Data;
    Data.Open(bpo::lmdb_folder_path, NEW);
    // Создаём объект, отвечающий за запись в базу данных
    LMDBTransaction* transaction = Data.NewTransaction();

    // просматриваем папки с изображениями и ищем все пути для изображений в формате *bmp
    QString dataFolder = QString::fromStdString(bpo::data_folder_path);
    QDirIterator it(dataFolder, QStringList() << "*.bmp", QDir::Files, QDirIterator::Subdirectories);
    std::vector<string> files_paths;

    // преобразуем в вектор с путями к изображениям и dat файлам
    while (it.hasNext())
    {
        it.next();
        files_paths.push_back(it.fileInfo().absolutePath().toStdString() + "/" + it.fileInfo().baseName().toStdString());
    }

    // перемешиваем случайным образом
    random_shuffle(files_paths.begin(),files_paths.end());

    // переменная для подсчёта кол-ва изображений в ОП
    int count = 0;

    // строка с данными и вектор с точками
    std::string matWithVecLabelAsStr, tmpstr;
    std::vector<std::vector<short>> points;

    cout << "Images found: " << files_paths.size() << endl;
    for (int i = 0; i < files_paths.size(); ++i)
    {
        // загружаем изображение
        cv::Mat img = cv::imread(files_paths[i] + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);

        // читаем точки из .dat файла
        dlib::deserialize(files_paths[i] + ".dat") >> points;

        // сериализуем все это дело в строчку
        serialize8UMatWithVecLabelToStr(img, points, matWithVecLabelAsStr);

        count++;
        // Ключ в виде "0000000001", "0000002384", и т.д.
        tmpstr = std::to_string(i);
        int length = 10 - tmpstr.length();
        for (int s = 0; s < length; ++s)
        {
            tmpstr.insert(0,"0");
        }

        // добавляем полученную строку с ключом в транзакцию
        transaction->Put(tmpstr, matWithVecLabelAsStr);

        // когда накопилось пять или закончились изображения, то добавляем накопленные в транзакции изображения в базу
        if (count >= 1000 || (i >= files_paths.size() - 1))
        {
            transaction->Commit();
            count = 0;
            cout << std::to_string(i) << " | " << files_paths[i] << " Added to database!" << endl << endl;
        }
    }
    cout << "Filling complete" << endl;
    Data.Close();
}
