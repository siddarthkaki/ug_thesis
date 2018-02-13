/**
 * @file Main
 * @brief 
 * @author S. Kaki
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Vector3d;

void readme();

std::map<std::string,std::string> LoadConfig(std::string filename);

Rect BoundingBox(Mat *img_cam, std::vector<KeyPoint> *keypoints);

std::vector< std::vector<KeyPoint> > DBSCAN_keypoints(std::vector<KeyPoint> *keypoints, float eps, int min_pts);
std::vector<int> RegionQuery(std::vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps);

void ColmapSiftFeatures(std::map<std::string,std::string> config_params, std::string cm_database_path);
std::vector<cv::KeyPoint> ColmapSiftKeypoints(std::string cm_database_path);
cv::Mat ColmapSiftDescriptors(std::string cm_database_path);

/**
 * @function main
 * @brief main function
 */
int main( int argc, char** argv )
{
    // start clock
    clock_t t;
    t = clock();

    //-- loading data --///////////////////////////////////////////////////////

    Reconstruction recon;

    // input: serialised file
    if( argc == 2 ) 
    {
        recon = recon.Load(argv[1]);
    }

    // inputs: points3D.txt and database.db
    else if( argc == 3 )
    {
        // new Reconstruction object
        Reconstruction temp(argv[1], argv[2]);

        recon = temp;

        // load data from points3D.txt and database.db
        recon.Load();
    }

    // inputs: points3D.txt, database.db, and file to save serialised data to
    else if( argc == 4 )
    {
        // new Reconstruction object
        Reconstruction temp(argv[1], argv[2]);

        recon = temp;

        // load data from points3D.txt and database.db
        recon.Load();
        recon.Save(argv[3], recon);
    }

    else { readme(); return -1; }

    // stop clock
    t = clock() - t;
    printf ("Loading time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);



    //-- projection --/////////////////////////////////////////////////////////

    // start clock
    t = clock();

    // camera parameters
    unsigned image_size_x = 1000, focal_length_x = 500, 
             image_size_y = 1000, focal_length_y = 500,
             skew = 0;

    // camera intrinsics matrix
    cv::Mat cam_mat = (Mat_<double>(3,3) <<  
                               focal_length_x,           skew, image_size_x/2,
                                            0, focal_length_y, image_size_y/2,
                                            0,              0,              1);

    // IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    // 1 0.0463085 0.0926584 0.817966 0.565863 -0.743449 -3.23341 7.8009 1 frame00038.jpg

    // DCM for camera extrinsics
    cv::Mat rot_mat = (Mat_<double>(3,3) <<
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1);
    cv::Mat rvec;
    Rodrigues(rot_mat, rvec);
    
    //std::cout << rvec << std::endl;

    // translation vector for camera extrinsics   
    cv::Mat tvec = (Mat_<double>(3,1) << -0.743449, -3.23341, 7.8009);

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

    // stop clock
    t = clock() - t;
    printf ("Projection time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);


    //-- auto_cropper --///////////////////////////////////////////////////////

    // start clock
    t = clock();

    // read in config file - TODO read from command argument
    std::map<std::string,std::string> config_params = LoadConfig("config/brickseal1.ini");

    // read in camera image to compare to map
    cv::Mat img_cam = imread( config_params.at("img_cam"), CV_LOAD_IMAGE_GRAYSCALE );

    if( !img_cam.data ) { printf(" --(!) Error reading image \n"); return -1; }

    // read in threshold for discriptor "distance" comparison
    float threshold = atof( config_params.at("feature_comparison_max_distance").c_str() );

    //-- Step 1: Detect the keypoints using SURF/SIFT Detector
    int min_hessian = atoi( config_params.at("min_hessian").c_str() );

    SiftFeatureDetector detector( min_hessian );
    // TODO - check if keypoints_map is actually needed
    //std::vector<cv::KeyPoint> keypoints_cam, keypoints_map;
    //detector.detect( img_cam, keypoints_cam );

    //-- Step 2: Calculate descriptors (feature vectors)
    SiftDescriptorExtractor extractor;
    //cv::Mat descriptors_cam;
    //extractor.compute( img_cam, keypoints_cam, descriptors_cam );

    std::string cm_database_path = "brickseal1_img_cam.db";
    ColmapSiftFeatures(config_params, cm_database_path);
    std::vector<cv::KeyPoint> keypoints_cam = ColmapSiftKeypoints(cm_database_path);
    cv::Mat descriptors_cam = ColmapSiftDescriptors(cm_database_path);

    // convert Eigen matrix of mean descriptors to CV matrix
    cv::Mat descriptors_map;
    eigen2cv(recon.mean_descriptors, descriptors_map);
    //eigen2cv(recon.all_descriptors, descriptors_map);
    //cv::Mat vlf_descriptors_map = VLFeatSiftFeatures(img_cam);

    std::cout << "COLMAP Cam SIFT Size: " << descriptors_cam.size() << std::endl;
    std::cout << "COLMAP Map SIFT Size: " << descriptors_map.size() << std::endl;
    //std::cout << "   VLF Map SIFT Size: " << vlf_descriptors_map.size() << std::endl;


    //std::cout << "Cam Descriptors:\n" << descriptors_cam << std::endl;
    //std::cout << "Map Descriptors:\n" << descriptors_map << std::endl;

    // convert descriptor matrices to CV_32F format if needed
    if( descriptors_cam.type() != CV_32F )
    { descriptors_cam.convertTo(descriptors_cam, CV_32F); }
    if( descriptors_map.type() != CV_32F )
    { descriptors_map.convertTo(descriptors_map, CV_32F); }
    //if( vlf_descriptors_map.type() != CV_32F )
    //{ vlf_descriptors_map.convertTo(vlf_descriptors_map, CV_32F); }

    //-- Step 3: Matching camera and map descriptors using FLANN matcher
    FlannBasedMatcher matcher;
    //BFMatcher matcher(NORM_L2);
    std::vector< DMatch > matches;
    matcher.match( descriptors_cam, descriptors_map, matches );

    double max_dist = 0; double min_dist = 1000;

    //-- compute the maximum and minimum distances between matched keypoints
    for( int i = 0; i < descriptors_cam.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<DMatch> good_matches;
    std::vector<DMatch>  bad_matches;

    for( int i = 0; i < descriptors_cam.rows; i++ )
    {
        if( matches[i].distance <= max(threshold*min_dist, 0.02) )
        { good_matches.push_back( matches[i] ); }
        else { bad_matches.push_back( matches[i] ); }
    }

    //-- find matched keypoints
    std::vector<KeyPoint> obj_keypts = keypoints_cam; // keypoints of changed object / umnatched keypoints
    std::vector<KeyPoint> matched_keypts; // keypoints of matched objects

    for( int i = good_matches.size()-1; i >= 0; i-- )
    {
        matched_keypts.push_back( keypoints_cam[ good_matches[i].queryIdx ] );
        obj_keypts.erase( obj_keypts.begin() + good_matches[i].queryIdx );
    }

    //-- cluster remaining unmatched keypoints
    float eps = atof( config_params.at("cluster_eps").c_str() );;
    int min_pts = atoi( config_params.at("cluster_min_pts").c_str() );
    std::vector< std::vector<KeyPoint> > point_clusters = DBSCAN_keypoints( &obj_keypts, eps, min_pts );


    //-- debug output
    printf("\n----Total Keypoints : %lu\n", keypoints_cam.size());
    printf("--Matched Keypoints : %lu\n", matched_keypts.size());
    printf("---Object Keypoints : %lu\n\n", obj_keypts.size());


    //-- image display output
    if(1)
    {
        Mat out_img_1, out_img_2, out_img_3;

        drawKeypoints(img_cam, obj_keypts, out_img_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "Object localisation", out_img_1 );

        drawKeypoints(img_cam, matched_keypts, out_img_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "Matched Keypoints", out_img_2 );

        drawKeypoints(img_cam, keypoints_cam, out_img_3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "All Keypoints", out_img_3 );
    }




    Mat img_cam_rgb = imread( config_params.at("img_cam"), CV_LOAD_IMAGE_COLOR ); // camera image in colour

    std::string output_dir = config_params.at("output_dir").c_str();

    std::string dataset_id = config_params.at("id").c_str();

    //-- Clustered KeyPoints image display output
    for (int i = 0; i < point_clusters.size(); i++)
    {

        std::vector<KeyPoint> current_cluster = point_clusters[i];
        
        int current_cluster_size = current_cluster.size();

        printf("Cluster:%d\tSize:%d\n", i, current_cluster_size);

        int min_size = atoi( config_params.at("cluster_min_size").c_str() );

        if (current_cluster_size > min_size)
        {

            Rect ROI = BoundingBox( &img_cam, &current_cluster );

            //Mat img_cam_cropped = img_cam;
            Mat img_cam_cropped = img_cam_rgb( ROI );

            Mat out_img_temp;
            drawKeypoints( img_cam_cropped, current_cluster, out_img_temp, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            
            std::ostringstream oss;
            oss << "Cluster:" << i;
            std::string title_text = oss.str();
            imshow( title_text, out_img_temp );
            imwrite( output_dir+dataset_id+"_"+std::to_string(i)+".jpg", out_img_temp );
        }
    }

    // stop clock
    t = clock() - t;
    printf ("Cropping time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);

    waitKey(0);
    return 0;
}



/******************************** FUNCTIONS **********************************/

/**
 * @function readme
 */
void readme()
{
    printf("Usage:\n");
    printf(" ./main <path_to_points3D.txt> <path_to_database.db>\n");
    printf(" ./main <path_to_points3D.txt> <path_to_database.db> <path_to_save_serialised_data.txt>\n");
    printf(" ./main <path_to_load_serialised_data.txt>\n");

}

/**
 * @function LoadConfig
 * @brief loads config params from ini file
 * @return map of key-value pairs
 */
std::map<std::string,std::string> LoadConfig(std::string filename)
{
    std::ifstream input(filename.c_str()); // the input stream
    std::map<std::string,std::string> out; // a map of key-value pairs in the file
    while(input) // keep on going as long as the file stream is good
    {
        std::string key; // the key
        std::string value; // the value
        std::getline(input, key, '='); // read up to the : delimiter into key
        std::getline(input, value, '\n'); // read up to the newline into value
        std::string::size_type pos1 = value.find_first_of("\""); // find the first quote in the value
        std::string::size_type pos2 = value.find_last_of("\""); // find the last quote in the value
        
        value = value.substr(pos1+1,pos2-pos1-1); // take a substring of the part between the quotes
        out[key] = value; // store the result in the map
        /*
        if(pos1 != std::string::npos && pos2 != std::string::npos && pos2 > pos1) // check if the found positions are all valid
        {
            value = value.substr(pos1+1,pos2-pos1-1); // take a substring of the part between the quotes
            out[key] = value; // store the result in the map
        }
        else
        {
            printf("invalid field\n");
        }
        */
    }
    input.close(); // close the file stream
    return out; // and return the result
}

/**
 * @function BoundingBox
 * @brief provides bounding box corners for a set of keypoints
 */
Rect BoundingBox(Mat *img_cam, std::vector<KeyPoint> *keypoints)
{
    float min_x = img_cam->cols, max_x = 0.0, min_y = img_cam->rows, max_y = 0.0;

    for (int i = 0; i < keypoints->size(); i++)
    {
        if (keypoints->at(i).pt.x < min_x) { min_x = keypoints->at(i).pt.x; }
        if (keypoints->at(i).pt.x > max_x) { max_x = keypoints->at(i).pt.x; }
        if (keypoints->at(i).pt.y < min_y) { min_y = keypoints->at(i).pt.y; }
        if (keypoints->at(i).pt.y > max_y) { max_y = keypoints->at(i).pt.y; }
    }

    /*
    printf("Cols: %d\tRows:%d\n", img_cam->cols, img_cam->rows);
    printf("%f\t%f\t%f\t%f\n", min_x, max_x, min_y, max_y);
    printf("%f\t%f\t%f\t%f\n", min_x, min_y, fabs(max_x - min_x), fabs(max_y - min_y));
    */

    Rect ROI(min_x, min_y, fabs(max_x - min_x), fabs(max_y - min_y));

    return ROI;
}

/**
 * @function DBSCAN_keypoints
 * @brief density-based spatial clustering of applications with noise
 */
std::vector< std::vector<KeyPoint> > DBSCAN_keypoints(std::vector<KeyPoint> *keypoints, float eps, int min_pts)
{
    std::vector< std::vector<KeyPoint> > clusters;
    std::vector<bool> clustered;
    std::vector<int> noise;
    std::vector<bool> visited;
    std::vector<int> neighbor_pts;
    std::vector<int> neighbor_pts_;
    int c;

    int num_keys = keypoints->size();

    //init clustered and visited
    for(int k = 0; k < num_keys; k++)
    {
        clustered.push_back(false);
        visited.push_back(false);
    }

    c = 0;
    clusters.push_back(std::vector<KeyPoint>()); // will stay empty?

    //for each unvisited point P in dataset keypoints
    for(int i = 0; i < num_keys; i++)
    {
        if(!visited[i])
        {
            //Mark P as visited
            visited[i] = true;
            neighbor_pts = RegionQuery(keypoints, &keypoints->at(i), eps);
            if(neighbor_pts.size() < min_pts)
                //Mark P as Noise
                noise.push_back(i);
            else
            {
                clusters.push_back(vector<KeyPoint>());
                c++;
                //expand cluster
                clustered[i] = true;
                // add P to cluster c
                clusters[c].push_back(keypoints->at(i));
                //for each point P' in neighbor_pts
                for(int j = 0; j < neighbor_pts.size(); j++)
                {
                    //if P' is not visited
                    if(!visited[neighbor_pts[j]])
                    {
                        //Mark P' as visited
                        visited[neighbor_pts[j]] = true;
                        neighbor_pts_ = RegionQuery(keypoints,&keypoints->at(neighbor_pts[j]),eps);
                        if(neighbor_pts_.size() >= min_pts)
                        {
                            neighbor_pts.insert(neighbor_pts.end(),neighbor_pts_.begin(),neighbor_pts_.end());
                        }
                    }
                    // if P' is not yet a member of any cluster
                    // add P' to cluster c
                    if(!clustered[neighbor_pts[j]])
                    {
                        clustered[neighbor_pts[j]] = true;
                        clusters[c].push_back(keypoints->at(neighbor_pts[j]));
                    }
                }
            }
        }
    }
    return clusters;
}

/**
 * @function RegionQuery
 * @brief searching for closest keypoints to keypoint in question
 */
vector<int> RegionQuery(std::vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps)
{
    float dist;
    vector<int> ret_keys;
    for(int i = 0; i < keypoints->size(); i++)
    {
        dist = sqrt(pow((keypoint->pt.x - keypoints->at(i).pt.x),2) + pow((keypoint->pt.y - keypoints->at(i).pt.y), 2));
        if(dist <= eps && dist != 0.0f)
        {
            ret_keys.push_back(i);
        }
    }
    return ret_keys;
}

/**
 * @function ColmapSiftFeatures
 * @brief COLMAP implementation of SIFT feature extraction
 */
void ColmapSiftFeatures(std::map<std::string,std::string> config_params, std::string cm_database_path)
{
    // COLMAP SYS CALL
    std::string cm_image_path = config_params.at("img_cam");
    system("mkdir .temp_img");
    std::string sys_call = "cp " + cm_image_path + " .temp_img/";
    system(sys_call.c_str());
    //std::string cm_database_path = "brickseal1_img_cam.db";
    //std::cout << cm_database_path << std::endl;
    std::string cm_sys_call = "colmap feature_extractor --database_path " + cm_database_path + " --image_path .temp_img --SiftExtraction.use_gpu 0";
    std::cout << cm_sys_call << std::endl;
    int cm_sys_res = system(cm_sys_call.c_str());
}

/**
 * @function ColmapSiftKeypoints
 * @brief COLMAP implementation of SIFT keypoints
 */
std::vector<cv::KeyPoint> ColmapSiftKeypoints(std::string cm_database_path)
{
    // setup db connection
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    rc = sqlite3_open(cm_database_path.c_str(), &db);
    if( rc ) { fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db)); exit (EXIT_FAILURE); }
    else     { fprintf(stderr, "Opened database successfully\n"); }

    // keypoint extraction ////////////////////////////////////////////////////
    //unsigned char *pzBlob; // holder pointer to blob
    //int *pnBlob; // retrieved blob size

    sqlite3_stmt *stmt;
    std::string zSql = "SELECT * FROM keypoints";
    sqlite3_prepare_v2(db, zSql.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);

    std::vector<cv::KeyPoint> img_keypoints;

    if (rc == SQLITE_ROW)
    {
        unsigned num_keypoints = sqlite3_column_int(stmt,1); // get num of descriptors from column 1
        unsigned num_cols = sqlite3_column_int(stmt,2); // get num of descriptors from column 1

        // allocate space for entire descriptors matrix
        //img_descriptors.resize(curr_num_descriptors,128);

        // get float32 BLOB data from db
        std::vector<char> data( sqlite3_column_bytes(stmt, 3) );
        const char *pBuffer = reinterpret_cast<const char*>( sqlite3_column_blob(stmt, 3) );
        std::copy( pBuffer, pBuffer + data.size(), &data[0] );

        unsigned curr_start = 0; // checked

        for ( unsigned j = 0; j < num_keypoints; j++ )
        {
            cv::KeyPoint kp_temp;
            kp_temp.pt.x = 0.0;
            kp_temp.pt.y = 0.0;

            //kp_temp.pt.x = atof(&pBuffer[0+curr_start*4]);
            //kp_temp.pt.y = atof(&pBuffer[4+curr_start*4]);
            
            //memcpy( &kp_temp.pt.x, pBuffer[0+curr_start*4], sizeof(float) );
            //memcpy( &kp_temp.pt.y, pBuffer[4+curr_start*4], sizeof(float) );

            char temp_x[sizeof(float)];
            char temp_y[sizeof(float)];

            std::copy( pBuffer + curr_start*4    , pBuffer + curr_start*4 + 4, &temp_x[0] );
            std::copy( pBuffer + curr_start*4 + 4, pBuffer + curr_start*4 + 8, &temp_y[0] );

            memcpy(&kp_temp.pt.x, &temp_x, sizeof(float));
            memcpy(&kp_temp.pt.y, &temp_y, sizeof(float));

            //kp_temp.pt.x = (float)(temp_x);
            //kp_temp.pt.y = (float)(temp_y);
            
            //kp_temp.pt.x = ((float) data.at(curr_start + 0));
            //kp_temp.pt.y = ((float) data.at(curr_start + 1));

            img_keypoints.push_back(kp_temp);

            curr_start += 6;
        }
    }
    return img_keypoints;
}

/**
 * @function ColmapSiftDescriptors
 * @brief COLMAP implementation of SIFT descriptors
 */
cv::Mat ColmapSiftDescriptors(std::string cm_database_path)
{

    // setup db connection
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    rc = sqlite3_open(cm_database_path.c_str(), &db);
    if( rc ) { fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db)); exit (EXIT_FAILURE); }
    else     { fprintf(stderr, "Opened database successfully\n"); }

    // descriptor extraction //////////////////////////////////////////////////

    sqlite3_stmt *stmt;
    std::string zSql = "SELECT * FROM descriptors";
    sqlite3_prepare_v2(db, zSql.c_str(), -1, &stmt, NULL);
    rc = sqlite3_step(stmt);

    MatrixXi img_descriptors;

    if (rc == SQLITE_ROW)
    {
        unsigned curr_num_descriptors = sqlite3_column_int(stmt,1); // get num of descriptors from column 1

        // allocate space for entire descriptors matrix
        img_descriptors.resize(curr_num_descriptors,128);

        // get uint8_t BLOB data from db
        std::vector<char> data( sqlite3_column_bytes(stmt, 3) );
        const char *pBuffer = reinterpret_cast<const char*>( sqlite3_column_blob(stmt, 3) );
        std::copy( pBuffer, pBuffer + data.size(), &data[0] );

        unsigned curr_start = 0; // checked
        unsigned curr_end = curr_start+127; // checke

        for ( unsigned j = 0; j < curr_num_descriptors; j++ )
        {
            VectorXi curr_descriptor(128);
            for ( unsigned k = 0; k < 128; k++ )
            {
                curr_descriptor(k) = ( (int) ((uint8_t) data.at(curr_start + k)) );
            }
            img_descriptors.block<1,128>(j,0) = curr_descriptor;
            curr_start += 128;
        }
    }

    rc = sqlite3_close(db);

    cv::Mat descriptors_cam;
    eigen2cv(img_descriptors, descriptors_cam);

    return descriptors_cam;
}