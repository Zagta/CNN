#include "boost_program_options.h"

using bpo = boost_program_options;

bpo::boost_program_options() {}

// входные параметры
std::string bpo::data_folder_path;                                       // путь к данных
std::string bpo::lmdb_folder_path;                                       // путь к базе
std::string bpo::snapshot_directory_path;                                // директория для снимков
std::string bpo::background_directory_path;                              // директория для фонов
std::string bpo::network_name;                                           // имя сети
std::string bpo::iter_name;                                              // номер итерации для снапшота
int bpo::mini_batch_size;                                                // размер мини-пакета
float bpo::learning_rate;                                                // коэффициент обучения
float bpo::required_learning_rate;                                       // требуемый коэффициент обучения
float bpo::learning_rate_shrink;                                         // шаг уменьшения коэффициента обучения
int bpo::iteration_without_progress;                                     // число итераций без прогресса
float bpo::weight_decay;                                                 // уменьшение веса
bool bpo::backgrounds;                                                   // использование различных фонов
bool bpo::random_cropper_enabled;                                        // использование изменения изображения и повороты
bool bpo::random_cropper_resize;                                         // использование изменения размера изображения
bool bpo::random_cropper_flip;                                           // использование отражения изображения по горизонтали
bool bpo::eye_reshaper;                                                  // использование eye_reshaper
bool bpo::multiply_network;                                              // использование второй сети
bool bpo::check_image;                                                   // использование второй сети
int bpo::images;                                                         // 42100; Helen - 1605; celba4500 - 3600; celeba+helens - 4600; lfwp - 600; accurate300wDetect - 2000; accurate300WCalc - 12000; accurate300wPlusCalc - 18500; celebafull - 530000 (всего 547221);
int bpo::multiply_network_number;                                        // номер используемой сети
int bpo::eye_chance;                                                     // вероятность закрытия глаза
std::string bpo::draft_network_name;                                     // имя черновой сети
int bpo::cascade_num;                                                    // параметр для обучения каскада
std::string bpo::draft_iter_name;                                        // номер итерации для чернового снапшота
// размеры для патчей бровей, носа и глаз
dlib::matrix<int,2,5> bpo::patches_sizes;
// количество точек
dlib::matrix<int,1,5> bpo::points_sizes;


// функция задания параметров
void bpo::SetParams(po::variables_map &vm)
{

    patches_sizes = 80, 80, 70, 70, 70,
                    50, 50, 90, 50, 50;
    points_sizes = 5, 5, 9, 6, 6;

    if(vm.count("lr"))
    {
        learning_rate = vm["lr"].as<float>();
        cout << "Learning rate is: " << learning_rate << endl;
    }

    if(vm.count("rlr"))
    {
        required_learning_rate = vm["rlr"].as<float>();
        cout << "Required learning rate is: " << required_learning_rate << endl;
    }

    if(vm.count("sh"))
    {
        learning_rate_shrink = vm["sh"].as<float>();
        cout << "Shrink learning rate factor is: " << learning_rate_shrink << endl;
    }

    if(vm.count("iter"))
    {
        iteration_without_progress = vm["iter"].as<int>();
        cout << "Iterations without progress is: " << iteration_without_progress << endl;
    }

    if(vm.count("im"))
    {
        images = vm["im"].as<int>();
        cout << "Training set number: " << images << endl;
    }

    if(vm.count("bs"))
    {
        mini_batch_size = vm["bs"].as<int>();
        cout << "Batch size is: " << mini_batch_size << endl;
    }

    if(vm.count("wd"))
    {
        weight_decay = vm["wd"].as<float>();
        cout << "Weight decay is: " << learning_rate << endl;
    }

    if(vm.count("ldbf"))
    {
        lmdb_folder_path = vm["ldbf"].as<string>();
        cout << "LMDB folder path is: " << lmdb_folder_path << endl;
    }

    if(vm.count("dsf"))
    {
        data_folder_path = vm["dsf"].as<string>();
        cout << "Dataset folder path is: " << data_folder_path << endl;
    }

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

    if(vm.count("bcg"))
    {
        background_directory_path = vm["bcg"].as<string>();
        cout << "Backgrounds folder path is: " << background_directory_path << endl;
    }

    if(vm.count("bcga"))
    {
        backgrounds = vm["bcga"].as<bool>();
        cout << "Backgrounds: " << backgrounds << endl;
    }

    if(vm.count("rndc"))
    {
        random_cropper_enabled = vm["rndc"].as<bool>();
        cout << "Random_cropper: " << random_cropper_enabled << endl;
    }

    if(vm.count("rndcr"))
    {
        random_cropper_resize = vm["rndcr"].as<bool>();
        cout << "Random_cropper_resize: " << random_cropper_resize << endl;
    }

    if(vm.count("ersh"))
    {
        eye_reshaper = vm["ersh"].as<bool>();
        cout << "Eye_reshaper: " << eye_reshaper << endl;
    }
    if(vm.count("ershc"))
    {
        eye_chance = vm["ershc"].as<int>();
        cout << "Eye_reshaper chance is: " << eye_chance << endl;
    }
    if(vm.count("rndcf"))
    {
        random_cropper_flip = vm["rndcf"].as<bool>();
        cout << "Random_cropper_flip: " << random_cropper_flip << endl;
    }

    if(vm.count("multi"))
    {
        multiply_network = vm["multi"].as<bool>();
        cout << "Multiply networks: " << multiply_network << endl;
    }

    if(vm.count("check"))
    {
        check_image = vm["check"].as<bool>();
        cout << "Check image: " << check_image << endl;
    }

    if(vm.count("m_num"))
    {
        multiply_network_number = vm["m_num"].as<int>();
        cout << "Multiply network number: " << multiply_network_number << endl;
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
}

// функция получения строки с параметрами, хелп и проч.
int bpo::GetParams(int argc, char *argv[])
{
    po::options_description desc("Application options");
    desc.add_options()
        ("help", "Show help message")
        ("lr", po::value<float>()->required()->default_value(0.0005), "Set learning rate")
        ("rlr", po::value<float>()->required()->default_value(0.000001), "Set required learning rate")
        ("sh", po::value<float>()->required()->default_value(0.1), "Set shrink learning rate factor")
        ("iter", po::value<int>()->required()->default_value(10000), "Set iteration without progress count")
        ("bs", po::value<int>()->required()->default_value(64), "Set mini-batch size")
        ("wd", po::value<float>()->required()->default_value(0.0001), "Set weight decay rate")
        ("ldbf", po::value<std::string>()->required()->default_value("/home/ginseng/Projects/DataSet/LMDB/DataWHelen"), "Set LMDB directory")
        ("dsf", po::value<std::string>()->required()->default_value("/home/ginseng/Projects/DataSet/IMAGES"), "Set dataset directory")
        ("nn", po::value<std::string>()->required()->default_value("Test"), "Set network name")
        ("sd", po::value<std::string>()->required()->default_value("/home/ginseng/Projects/DataSet/Snapshots"), "Set snapshot directory")
        ("bcg", po::value<std::string>()->default_value("/home/ginseng/Projects/DataSet/Backgrounds"), "Set backgrounds directory")
        ("in", po::value<std::string>()->default_value("10000"), "Set snapshot iteration number")
        ("im", po::value<int>()->required(), "Set number of pictures in training set")
        ("bcga", po::value<bool>()->required()->default_value(0), "Set if you want to use backgrounds")
        ("rndc", po::value<bool>()->required()->default_value(0), "Set if you want to use random_cropper")
        ("rndcr", po::value<bool>()->required()->default_value(0), "Set if you want to use random_cropper_resize")
        ("ersh", po::value<bool>()->required()->default_value(0), "Set if you want to use eye_reshaper")
        ("ershc", po::value<int>()->required()->default_value(30), "Set chance to close eye")
        ("rndcf", po::value<bool>()->required()->default_value(0), "Set if you want to use random_cropper_flip")
        ("multi", po::value<bool>()->required()->default_value(0), "Set if you want to train multiply networks")
        ("check", po::value<bool>()->required()->default_value(0), "Set if you want to check minibatch image")
        ("m_num", po::value<int>()->required()->default_value(0), "Set number of multiply network")
        ("cascade", po::value<int>()->required()->default_value(0), "Set cascade number")
        ("draftnn", po::value<std::string>(), "Set draft network name")
        ("draftin", po::value<std::string>()->default_value("10000"), "Set draft snapshot iteration number")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    SetParams(vm);
    std::cout << std::endl;

    return 1;
}
