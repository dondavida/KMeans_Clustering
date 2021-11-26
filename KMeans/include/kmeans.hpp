#ifndef K_MEANS_HPP_
#define K_MEANS_HPP_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace ipb{

        /**
        * @brief
        * 1. Given cluster centroids i initialized in some way,
        * 2. For iteration t=1..T:
        * 1. Compute the distance from each point x to each cluster centroid ,
        * 2. Assign each point to the centroid it is closest to,
        * 3. Recompute each centroid as the mean of all points assigned to it,
        **
        @param image The input image to cluster.
        * @param k The size of the dictionary, ie, number of visual words.
        * @param max_iterations Maximum number of iterations before convergence.
        * @return cv::Mat One unique Matrix representing all the $k$-means(stacked).
        **/

        cv::Mat RunKmeans(const cv::Mat& image, int k, int max_iter);
        cv::Mat KMeansInitCentroids(const cv::Mat& image, int k);
        std::vector<int> FindClosestCentroids(const cv::Mat& image, const cv::Mat& centroids);
        cv::Mat ComputeCentroids(const cv::Mat& image, std::vector<int>& idx, int k);
        std::vector<int> FindIndexPosition(const std::vector<int>& idx, int n);
        cv::Mat ConvertImageToFloat(const cv::Mat& image);
        cv::Mat ConvertCentroidsToImage(const cv::Mat& image, const cv::Mat& centroids);
        cv::Mat ConvertCentroidToImage(const cv::Mat& centroids);
            
}

#endif // K_MEANS_HPP_