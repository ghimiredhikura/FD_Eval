#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <sys/stat.h>
#include <sys/time.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);


enum DATASET { AFW, PASCAL };
enum MODE {WEBCAM, BENCHMARK_EVALUATION, IMAGE, IMAGE_LIST};

typedef struct _tagOPENCVSSDesult {
    cv::Rect r;
    float score;
}OPENCVSSDResult;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

std::vector<OPENCVSSDResult> opencvSSDDetection(dnn::Net& net, cv::Mat frame)
{
    std::vector<OPENCVSSDResult> m_results;

    //! [Prepare blob]
    Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                  Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images
    //! [Prepare blob]

    //! [Set input blob]
    net.setInput(inputBlob, "data"); //set the network input
    //! [Set input blob]

    //! [Make forward pass]
    Mat detection = net.forward("detection_out"); //compute output
    //! [Make forward pass]

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
        int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
        int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
        int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

        Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));

        OPENCVSSDResult m_result;
        m_result.score = confidence;
        m_result.r = object;

        m_results.push_back(m_result);
    }

    for(size_t i=0;i<m_results.size();i++) {
        if(m_results[i].score > 0.1)
            cv::rectangle(frame, m_results[i].r, CV_RGB(255,0,0), 2);
    }

    return m_results;
}

void run_webcam(int webcam_id)
{
    String modelConfiguration = "../resnet-face/deploy.prototxt";
    String modelBinary = "../resnet-face/res10_300x300_ssd_iter_140000.caffemodel";

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    cv::VideoCapture cap(webcam_id);
    if(!cap.isOpened()) {
        cout << "fail to open webcam!" << endl;
        return;
    }

    cv::Mat image;
    while(true) {
    
        cap >> image;

        if(image.empty()) break;
        
        if (image.channels() == 4)
            cvtColor(image, image, COLOR_BGRA2BGR);

        double time_begin = what_time_is_it_now();
        //detect face by min_size(30)
        std::vector<OPENCVSSDResult> m_results = opencvSSDDetection(net, image);
        double time_now = what_time_is_it_now();
        double time_diff = time_now-time_begin;
        cv::imshow("image", image);

        cout << "OPENCVSSD FPS: " << 1/time_diff << endl;

        if (cv::waitKey(1) >= 0) break;     
    }
}

void run_afw_pascal(DATASET m_data, std::string dataset_path)
{
    std::string str_img_file;
    if(m_data == AFW) str_img_file = "../detections/AFW/afw_img_list.txt";
    else if (m_data == PASCAL) str_img_file = "../detections/PASCAL/pascal_img_list.txt";
    else return;

    cout << str_img_file << endl;

    std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

    std::vector<string> image_list;
    std::string imname;
    while(std::getline(inFile, imname)) 
    {
        image_list.push_back(imname);
    }

    std::string str_out_file;
    if(m_data == AFW) str_out_file = "../detections/AFW/opencvssd_afw_dets.txt";
    else if (m_data == PASCAL) str_out_file = "../detections/PASCAL/opencvssd_pascal_dets.txt";

    std::ofstream outFile(str_out_file.c_str());

    String modelConfiguration = "../resnet-face/deploy.prototxt";
    String modelBinary = "../resnet-face/res10_300x300_ssd_iter_140000.caffemodel";

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    // process each image one by one
    for(int i = 0; i < image_list.size(); i++)
    {
        std::string imname = image_list[i];
        std::string tempname = imname;

        if(m_data==AFW) imname = dataset_path+imname+".jpg";
        else if(m_data==PASCAL) imname = dataset_path+imname;

        cout << "processing image " << i+1 << "/" << image_list.size() << " [" << imname.c_str() << "]" << endl;

        cv::Mat image = cv::imread(imname);

        if (image.channels() == 4)
           cvtColor(image, image, COLOR_BGRA2BGR);

        std::vector<OPENCVSSDResult> m_results = opencvSSDDetection(net, image);

        for(int j=0;j<m_results.size();j++) {
            outFile << tempname << " " << m_results[j].score << " " << m_results[j].r.x << " "
                << m_results[j].r.y << " " << m_results[j].r.x+m_results[j].r.width << " " 
                << m_results[j].r.y+m_results[j].r.height << endl;      
        }           

        imshow("test", image);

        waitKey(1);
    }
    outFile.close();
}

void run_image(std::string image_path)
{
    String modelConfiguration = "../resnet-face/deploy.prototxt";
    String modelBinary = "../resnet-face/res10_300x300_ssd_iter_140000.caffemodel";

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]
    
    cv::Mat image = cv::imread(image_path.c_str());
    if(image.empty()) {
        cout << "Image not exist in the specified dir!";
        return; 
    } 

    if (image.channels() == 4)
       cvtColor(image, image, COLOR_BGRA2BGR);

    std::vector<OPENCVSSDResult> m_results = opencvSSDDetection(net, image);

    cv::imshow("image", image);
    waitKey(1);
    getchar();
}

void run_images(std::string image_path)
{
    std::vector<cv::String> img_list;
    glob(image_path+"*.jpg", img_list, false);

    String modelConfiguration = "../resnet-face/deploy.prototxt";
    String modelBinary = "../resnet-face/res10_300x300_ssd_iter_140000.caffemodel";

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    size_t count = img_list.size();
    for(size_t i=0;i<count;i++) 
    {
        cv::Mat image = cv::imread(img_list[i].c_str());
        if(image.empty()) {
            cout << "Image not exist in the specified dir!";
            return; 
        }

        if (image.channels() == 4)
            cvtColor(image, image, COLOR_BGRA2BGR); 
    
        std::vector<OPENCVSSDResult> m_results = opencvSSDDetection(net, image);

        cv::imshow("image", image);
        cv::waitKey(1);
    }
} 


const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325) "
                    "with ResNet-10 architecture to detect faces on camera/video/image.\n"
                    "More information about the training is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt\n"
                    ".caffemodel model's file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel\n"
                    ".prototxt file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt\n";

cv::String keys =
    "{ help h       | | Print help message. }"
    "{ mode m       | 0 | Select running mode: "
                        "0: WEBCAM - get image streams from webcam, "
                        "1: IMAGE - detect faces in single image, "
                        "2: IMAGE_LIST - detect faces in set of images, "
                        "3: BENCHMARK_EVALUATION - benchmark evaluation, results will be stored in 'detections' }"
    "{ webcam i     | 0 | webcam id, if mode is 0 }"
    "{ dataset d    | AFW | select dataset, if mode is 3:"
                           "AFW: afw dataset, "
                           "PASCAL: pascal dataset }"
    "{ path p       | | Path to image file or image list dir or benchmark dataset }";


int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    if(argc == 1 || parser.has("help")) 
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    MODE m_mode;
    int mode = parser.get<int>("mode");     

    if(mode == 1) m_mode = IMAGE;
    else if(mode == 2) m_mode = IMAGE_LIST;
    else if(mode == 3) m_mode = BENCHMARK_EVALUATION;
    else m_mode = WEBCAM;

    if(m_mode == WEBCAM)
    {
        int webcam_id = parser.get<int>("webcam");
        run_webcam(webcam_id);
    }
    else if(m_mode == IMAGE) 
    {
        std::string image_path = parser.get<String>("path");
        run_image(image_path);
    }
    else if(m_mode == IMAGE_LIST) 
    {
        std::string image_path = parser.get<String>("path");
        run_images(image_path); 
    }
    else if (m_mode == BENCHMARK_EVALUATION)
    {
        DATASET m_data;
        std::string dataset_name = parser.get<String>("dataset");
        cout << dataset_name << endl;
        if(dataset_name == "PASCAL") m_data = PASCAL;
        else if(dataset_name == "AFW") m_data = AFW; 

        std::string dataset_path = parser.get<String>("path");

        if(m_data == AFW || m_data == PASCAL)
            run_afw_pascal(m_data, dataset_path);
    }

    return 0;
} // main
