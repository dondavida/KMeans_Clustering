#include "kmeans.hpp"

cv::Mat ipb::ConvertImageToFloat(const cv::Mat& image){

    cv::Mat image_(image.rows * image.cols, image.channels(), CV_32F);
    // Grayscale Image
    if(image.channels() == 1){
        for(int r = 0; r < image.rows; ++r){
            for(int c = 0; c < image.cols; ++c){
                image_.at<float>(r * image.cols + c, 0) = image.at<uchar>(r, c);
            }
        }
    }
    // RGB Image
    if(image.channels() == 3){
        for(int r = 0; r < image.rows; ++r){
            for(int c = 0; c < image.cols; ++c){
                for(int z = 0; z < image.channels(); ++z){
                    image_.at<float>(r * image.cols + c, z) = image.at<cv::Vec3b>(r, c)[z];
                }
            }
        }
    }
    return image_;  // return [rows * cols, 3] for RGB, [rows * cols, 1] for GrayScale
}

std::vector<int> ipb::FindIndexPosition(const std::vector<int>& idx, int n){

    std::vector<int> indexes{};
    for(unsigned int i = 0; i < idx.size(); ++i){
        if(idx[i] == n){
            indexes.emplace_back(i);
        }
    }
    return indexes;
}

cv::Mat ipb::KMeansInitCentroids(const cv::Mat& image, int k){

    cv::Mat centroids = cv::Mat::zeros(k, image.cols, CV_32F);

    if(image.cols == 1){
        // Generate Random Index
        std::vector<int> idx;
        for(int i = 0; i < image.rows; ++i){
            idx.emplace_back(i) ;
        }
        srand(unsigned(time(NULL)));
        std::random_shuffle(idx.begin(), idx.end()); // re-shufffled vec to index
        // Take first k examples as centroid
        for(int r = 0; r < k; ++r){
            centroids.at<float>(r, 0) = image.at<float>(idx[r], 0);  // centroids = [k x 1]   
        }
    }

    if(image.cols == 3){
        // Generate Random Index
        std::vector<int> idx;
        for(int i = 0; i < image.rows; ++i){
            idx.emplace_back(i) ;
        }
        srand(unsigned(time(NULL)));
        std::random_shuffle(idx.begin(), idx.end()); // re-shufffled vec to index
        // Take first k examples as centroid
        for(int r = 0; r < k; ++r){
            for(int c = 0; c < image.cols; ++c){
                centroids.at<float>(r, c) = image.at<float>(idx[r], c); // centroids = [k x 3]
            }       
        }
    }
    //std::cout << centroids(cv::Range(0, k), cv::Range(0,1)) << std::endl;
    return centroids;
}

std::vector<int> ipb::FindClosestCentroids(const cv::Mat& image, const cv::Mat& centroids){

    cv::Mat_<float> ColSum;
    cv::Mat_<float> eucl_dist;
    const int k = centroids.rows;
    std::vector<int> idx;
    cv::Mat_<float> centroids_ = cv::Mat_<float>(centroids);
    cv::Mat_<float> image_ = cv::Mat_<float>(image);
    // find index of nearest neighbor
    for(int i = 0; i < image_.rows; ++i){
        cv::Mat_<float> Diff = centroids_ - (cv::repeat(image_.row(i), k, 1));
        cv::Mat_<float> Square = Diff.mul(Diff); // perform square operation
        cv::reduce(Square, ColSum, 1, CV_REDUCE_SUM, CV_32F);
        cv::sqrt(ColSum, ColSum);
        eucl_dist = ColSum;
        double minval, maxval;
        int minIdx[2] = {}, maxIdx[2] = {};
        cv::minMaxIdx(eucl_dist, &minval, &maxval, minIdx, maxIdx);
        idx.emplace_back(minIdx[0]);
    }
    return idx;
}

cv::Mat ipb::ComputeCentroids(const cv::Mat& image, std::vector<int>& idx, int k){
    
    cv::Mat centroids = cv::Mat::zeros(k, image.cols, CV_32F);

    // Centroids for grayscale image
    if(image.cols == 1){
        cv::Mat SumImageMatrix;
        for(int i = 0; i < k; ++i){
            std::vector<int> indexes = FindIndexPosition(idx, i);
            int idx_count =  std::count(idx.begin(), idx.end(), i);
            cv::Mat image_ = cv::Mat::zeros(indexes.size(), 1, CV_32F);
            if(idx_count >= 1){
                for(unsigned int r = 0; r < indexes.size(); ++r){
                    image_.at<float>(r, 0) = image.at<float>(indexes[r], 0); //descriptor[row*col x 1]
                }
                cv::reduce(image_, SumImageMatrix, 0, CV_REDUCE_SUM, CV_32F);
            }
            cv::Mat result = SumImageMatrix / idx_count;
            centroids.at<float>(i, 0) = result.at<float>(0, 0);
        }
    }

    // centroids for RGB image
    if(image.cols == 3){
        cv::Mat SumImageMatrix;
        for(int i = 0; i < k; ++i){
            std::vector<int> indexes = FindIndexPosition(idx, i);
            int idx_count =  std::count(idx.begin(), idx.end(), i);
            cv::Mat image_ = cv::Mat::zeros(indexes.size(), image.cols, CV_32F); //descriptor[row*col x 3]
            if(idx_count >= 1){
                for(unsigned int r = 0; r < indexes.size(); ++r){
                    for (int c = 0; c < image.cols; ++c){
                        image_.at<float>(r, c) = image.at<float>(indexes[r], c);
                    }
                }
                cv::reduce(image_, SumImageMatrix, 0, CV_REDUCE_SUM, CV_32F);
            }
            cv::Mat result = SumImageMatrix / idx_count;
            for(int j = 0; j < image_.cols; ++j){
                centroids.at<float>(i, j) = result.at<float>(0,j);
            }
        }
    }    
    return centroids;
}

cv::Mat ipb::RunKmeans(const cv::Mat& image, int k, int max_iter){
  
    std::vector<cv::Mat> StackedCentroids;
    auto image_ = ConvertImageToFloat(image);
    cv::Mat centroids = KMeansInitCentroids(image_, k);
    for(int i = 0; i < max_iter; ++i){
        std::vector<int> idx = FindClosestCentroids(image_, centroids);
        auto centroids = ComputeCentroids(image_, idx, k);
        StackedCentroids.emplace_back(centroids);
    } 
    cv::Mat centroids_ = StackedCentroids[max_iter - 1];
    return centroids_;
}

cv::Mat ipb::ConvertCentroidsToImage(const cv::Mat& image, const cv::Mat& centroids){
    
    cv::Mat NewImage = cv::Mat::zeros(image.rows, image.cols, image.type());

    // Convert centroids[k x 1] to grayscale image
    if(image.channels() == 1){
        auto image_ = ConvertImageToFloat(image);
        std::vector<int> idx = FindClosestCentroids(image_, centroids);
	    for (int r = 0; r < image.rows; ++r){
		    for (int c = 0; c < image.cols; ++c){
			    NewImage.at<uchar>(r, c) = centroids.at<float>(idx[r * image.cols + c], 0);  // return compressed GrayScale image
	        }
        }
    }

    // convert centroids[k x 3] to RGB image
    if(image.channels() == 3){
        auto image_ = ConvertImageToFloat(image);
        std::vector<int> idx = FindClosestCentroids(image_, centroids);
	    for (int r = 0; r < image.rows; ++r){
		    for (int c = 0; c < image.cols; ++c){
			    int cluster_idx = idx[r * image.cols + c];
			    for (int z = 0; z < image.channels(); ++z) {
				    NewImage.at<cv::Vec3b>(r, c)[z] = centroids.at<float>(cluster_idx, z);  // return compressed RGB image
			    }
	        }
        }
    }
    return NewImage;
}

cv::Mat ipb::ConvertCentroidToCluster(const cv::Mat& centroids){
    
    // convert centroids value [k x 1] to color image
    if(centroids.cols == 1){
        cv::Mat centroids_ = cv::Mat::zeros(centroids.rows, 1, CV_8UC1);
        for(int i = 0; i < centroids.rows; ++i){
            centroids_.at<uchar>(i, 0) = centroids.at<float>(i, 0);
        }
        return centroids_;     
    }
    // convert centroids value [k x 3] to color image
    if(centroids.cols == 3){
        cv::Mat centroids_ = cv::Mat::zeros(centroids.rows, 1, CV_8UC3);
        for(int r = 0; r < centroids.rows; ++r){
            for(int z = 0; z < 3; ++z){
                centroids_.at<cv::Vec3b>(r, 0)[z] = centroids.at<float>(r, z);
            }
        }
        return centroids_;
    }
}


