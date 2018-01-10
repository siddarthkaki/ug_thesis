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

    // read in points3D.txt file name
    std::string points_file_id = argv[1];

    // read in database file name
    std::string db_file_id = argv[2];

    // setup db connection
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    rc = sqlite3_open(db_file_id.c_str(), &db);
    if( rc ) { fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db)); return(0); }
    else     { fprintf(stderr, "Opened database successfully\n"); }
    //sqlite3_close(db);

    unsigned char *pzBlob; // holder pointer to blob
    int *pnBlob; // retrieved blob size

    // number of 3D points in file
    unsigned num_points = LineCount(points_file_id) - 3;

    printf("Num 3D points: %u\n", num_points);

    MatrixXd pos_mat(num_points,3); // matrix of 3D position for each point
    MatrixXi mean_descriptors(num_points,128); // matrix of mean 128-length SIFT descriptors for each point

    std::ifstream input_file(points_file_id);
    input_file.seekg(std::ios::beg);
    std::string curr_line;

    if ( input_file.is_open() )
    {
        // skip header comments
        for( unsigned i = 0; i < 3; i++ ) { getline( input_file, curr_line ); }

        // loop through each point in 3D point cloud
        for( unsigned i = 0; i < num_points; i++ )
        {
            //if (i % 1000 == 0) { std::cout << i << std::endl; }
            //if (i % 9213 == 0) { std::cout << "Here" << std::endl; }

            getline( input_file, curr_line );

            // tokenise string into vector of substrings by ' ' delimiter
            std::istringstream iss( curr_line );
            std::vector<std::string> tokenised_string{std::istream_iterator<std::string>{iss},
                        std::istream_iterator<std::string>{}};

            // transform string vector into double vector
            std::vector<double> tokenised_double( tokenised_string.size() );
            std::transform( tokenised_string.begin(), tokenised_string.end(), tokenised_double.begin(),
                        []( const std::string& val ) { return std::stod(val); } );

            unsigned curr_id = tokenised_double.at(0); // POINT3D_ID
            
            pos_mat(i,0) = tokenised_double.at(1); // X pos
            pos_mat(i,1) = tokenised_double.at(2); // Y pos
            pos_mat(i,2) = tokenised_double.at(3); // Z pos

            //if (i % 9213 == 0) { std::cout << "Here" << std::endl; }

            unsigned num_tracks = (tokenised_double.size() - 8)/2; // number of images with this point

            //if (i % 1000 == 0) { std::cout << "Num Images: " << num_tracks << std::endl << std::endl; }
    
            MatrixXi curr_descriptors(num_tracks,128); // matrix of descriptors for each point

            // loop through each image with this point
            for( unsigned j = 0; j < num_tracks; j++ )
            {
                //if (i % 9213 == 0) { std::cout << "Here: j = " << j << std::endl; }
                //if (i % 1000 == 0) { std::cout << "Image: " << j+1 << std::endl; }
                unsigned curr_image_id = tokenised_double.at(6+2*(j+1)); // IMAGE_ID
                unsigned curr_point2d_idx = tokenised_double.at(7+2*(j+1)); // POINT2D_IDX

                //if (i % 9213 == 0) { std::cout << "Current IMAGE_ID: " << curr_image_id << std::endl; }
                //if (i % 9213 == 0) { std::cout << "Current POINT2D_IDX: " << curr_point2d_idx << std::endl; }

                sqlite3_stmt *stmt;
                std::string zSql = "SELECT * FROM descriptors WHERE image_id=" + std::to_string(curr_image_id);
                sqlite3_prepare_v2(db, zSql.c_str(), -1, &stmt, NULL);
                rc = sqlite3_step(stmt);
                
                if (rc == SQLITE_ROW)
                {
                    unsigned curr_num_descriptors = sqlite3_column_int(stmt,1); // get num of descriptors from column 1

                    // get uint8_t BLOB data from db
                    std::vector<char> data( sqlite3_column_bytes(stmt, 3) );
                    const char *pBuffer = reinterpret_cast<const char*>( sqlite3_column_blob(stmt, 3) );
                    std::copy( pBuffer, pBuffer + data.size(), &data[0] );

                    unsigned curr_start = (curr_point2d_idx-1)*128; // checked
                    unsigned curr_end = curr_start+127; // checked

                    //if (i % 9213 == 0) { std::cout << "BLOB Start Index: " << curr_start << std::endl; }
                    
                    //if (i % 9213 == 0) { std::cout << "BLOB Size: " << data.size() << std::endl
                    //                               << "Num Descriptors: " << data.size()/128 << std::endl; }

                    VectorXi curr_descriptor(128);

                    for ( unsigned k = 0; k < 128; k++ )
                    {
                        //if (i % 9213 == 0) { std::cout << "Here: k = " << k << std::endl; }
                        curr_descriptor(k) = ( (int) ((uint8_t) data.at(curr_start + k)) );
                        //if (i % 1000 == 0) { std::cout << curr_descriptor(k) << ' '; }
                    }
                    //if (i % 1000 == 0) { std::cout << std::endl << std::endl; }

                    curr_descriptors.block<1,128>(j,0) = curr_descriptor;
                }

                rc = sqlite3_finalize(stmt);

            }

            VectorXi mean_descriptor(128);
            mean_descriptor = curr_descriptors.colwise().mean();
            mean_descriptors.block<1,128>(i,0) = mean_descriptor;
            //if (i % 1000 == 0) { std::cout << mean_descriptor.transpose() << std::endl; }
            if (i % 5000 == 0) { std::cout << "Completed point: " << i << std::endl; }
        }

        //std::cout << mean_descriptors << std::endl;
    }

    rc = sqlite3_close(db);

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
    eigen2cv(pos_mat, pos_mat_cv);

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