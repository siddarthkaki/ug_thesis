/**
 * @file Main
 * @brief 
 * @author S. Kaki
 */

#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <map>
#include <eigen3/Eigen/Dense>
#include <sqlite3.h>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Reconstruction.h"

using namespace cv;

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Vector3d;

void readme();
unsigned LineCount(std::string filename);

/**
 * @function main
 * @brief main function
 */
int main( int argc, char** argv )
{
    if( argc != 3 )
    { readme(); return -1; }

    // start clock
    clock_t t;
    t = clock();

    // new Reconstruction object
    Reconstruction recon(argv[1], argv[2]);

    // load data from points3D.txt and database.db
    recon.Load();
    //recon.Save(recon, "brick_seal_serial.txt");

    //-- projection --/////////////////////////////////////////////////////////

    // camera parameters
    unsigned image_size_x = 1000, focal_length_x = 500, 
             image_size_y = 1000, focal_length_y = 500,
             skew = 0;

    // camera intrinsics matrix
    cv::Mat cam_mat = (Mat_<double>(3,3) <<  
                               focal_length_x,           skew, image_size_x/2,
                                            0, focal_length_y, image_size_y/2,
                                            0,              0,              1);

    // DCM for camera extrinsics
    cv::Mat rot_mat = (Mat_<double>(3,3) <<
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1);
    cv::Mat rvec;
    Rodrigues(rot_mat, rvec);
    
    //std::cout << rvec << std::endl;

    // translation vector for camera extrinsics   
    cv::Mat tvec = (Mat_<double>(3,1) << -1, -4, 5);

    // distortion cooefficients vector; no distortion
    cv::Mat dist_coeffs;

    // matrix for projected points
    cv::Mat proj_pos;

    // convert Eigen matrix of 3D point cloud to CV matrix
    cv::Mat pos_mat_cv;
    eigen2cv(recon.pos_mat, pos_mat_cv);

    // project 3D point cloud to 2D camera view
    projectPoints(pos_mat_cv, rvec, tvec, cam_mat, dist_coeffs, proj_pos);

    //std::cout << proj_pos << std::endl;
    //waitKey(0);

    // stop clock
    t = clock() - t;
    printf ("Run time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);

    return 0;
}




/******************************** FUNCTIONS **********************************/

/**
 * @function readme
 */
void readme()
{ printf(" ./main <path_to_points3D.txt> <path_to_database.db>\n"); }

/**
 * @function LineCount
 * @brief counts number of lines in a file
 * @return unsigned int containings number of lines
 */
unsigned LineCount(std::string filename)
{
    std::ifstream input_file(filename);

    // new lines will be skipped unless we stop it from happening
    input_file.unsetf(std::ios_base::skipws);

    // count the newlines
    unsigned count = std::count(
        std::istream_iterator<char>(input_file),
        std::istream_iterator<char>(), 
        '\n');

    return count;
}