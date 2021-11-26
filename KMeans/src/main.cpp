#include "kmeans.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(){
    // png image
    const std::string filename = "D:/Users from C drive/modern_c++_Vizzo/KMeans/data/imageCompressedCam0_0000700.png";
    cv::Mat ImageRGB = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat ImageGray;
    cv::cvtColor(ImageRGB,ImageGray,cv::COLOR_BGR2GRAY);
    int k = 3;
    int max_iter = 10;
    auto centroids = ipb::RunKmeans(ImageRGB, k, max_iter);
    //auto centroids = ipb::RunKmeans(ImageGray, k, max_iter);
    std::cout << centroids << std::endl;
    auto compressed_image = ipb::ConvertCentroidsToImage(ImageRGB, centroids);
    //auto compressed_image = ipb::ConvertCentroidsToImage(ImageGray, centroids);
    auto color_cluster = ipb::ConvertCentroidToCluster(centroids);
    std::string window_name = "compressed image";
    std::string cluster_name = "cluster RGB";
    //cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::namedWindow(cluster_name, cv::WINDOW_NORMAL);
    cv::imshow(cluster_name, color_cluster);
    //cv::imshow(window_name, compressed_image);
    cv::waitKey();
    return 0;
}