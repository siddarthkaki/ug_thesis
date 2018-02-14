#ifndef COLMAPIMG_H
#define COLMAPIMG_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <sqlite3.h>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"

using Eigen::MatrixXi;

class ColmapImg
{
    private:
        std::string cm_database_path;
        std::map<std::string,std::string> config_params;
        std::vector<cv::KeyPoint> img_keypoints;
        MatrixXi img_descriptors;
        cv::Mat descriptors_cam;

    public:
        ColmapImg(std::map<std::string,std::string> arg_config_params);
        void ColmapSiftFeatures();
        std::vector<cv::KeyPoint> ColmapSiftKeypoints();
        cv::Mat ColmapSiftDescriptors();
        void ColmapClean();

};

#endif