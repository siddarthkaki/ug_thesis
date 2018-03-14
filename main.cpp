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
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/xfeatures2d/nonfree.hpp"

#include "Reconstruction.h"
#include "ColmapImg.h"

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
    std:string config_params_str;

    // input: serialised file & config params file
    if( argc == 3 ) 
    {
        recon = recon.Load(argv[1]);
        config_params_str = argv[2];
    }

    // inputs: points3D.txt, database.db, & config params file
    else if( argc == 4 )
    {
        // new Reconstruction object
        Reconstruction temp(argv[1], argv[2]);
        config_params_str = argv[3];

        recon = temp;

        // load data from points3D.txt & database.db
        recon.Load();
    }

    // inputs: points3D.txt, database.db, file to save serialised data to,
    //         & config params file
    else if( argc == 5 )
    {
        // new Reconstruction object
        Reconstruction temp(argv[1], argv[2]);

        recon = temp;

        // load data from points3D.txt and database.db
        recon.Load();
        recon.Save(argv[3], recon);

        config_params_str = argv[4];
    }

    else { readme(); return -1; }

    // stop clock
    t = clock() - t;
    printf ("Loading time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);


    //-- feature correlation --////////////////////////////////////////////////

    // start clock
    t = clock();

    // read in config file
    std::map<std::string,std::string> config_params = LoadConfig(config_params_str);

    // read in camera image to compare to map
    cv::Mat img_cam = imread( config_params.at("img_cam"), CV_LOAD_IMAGE_GRAYSCALE );
    if( !img_cam.data ) { printf(" --(!) Error reading image \n"); return -1; }

    // read in threshold for discriptor "distance" comparison
    float threshold = atof( config_params.at("feature_comparison_max_distance").c_str() );

    // read in ratio value for ratio test
    float RATIO = atof( config_params.at("ratio").c_str() );

    // process new camera image with COLMAP
    ColmapImg cm_img(config_params);
    cm_img.ColmapSiftFeatures();
    std::vector<cv::KeyPoint> keypoints_cam = cm_img.ColmapSiftKeypoints();
    cv::Mat descriptors_cam = cm_img.ColmapSiftDescriptors();
    cm_img.ColmapClean();

    // convert Eigen matrix of mean descriptors to CV matrix
    cv::Mat descriptors_map;
    eigen2cv(recon.point_descriptors, descriptors_map);

    // output number of descriptors
    std::cout << "COLMAP Cam SIFT Size: " << descriptors_cam.size() << std::endl;
    std::cout << "COLMAP Map SIFT Size: " << descriptors_map.size() << std::endl;

    // convert descriptor matrices to CV_32F format if needed
    if( descriptors_cam.type() != CV_32F )
    { descriptors_cam.convertTo(descriptors_cam, CV_32F); }
    if( descriptors_map.type() != CV_32F )
    { descriptors_map.convertTo(descriptors_map, CV_32F); }

    //-- match camera and map descriptors using FLANN matcher
    FlannBasedMatcher matcher;

    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::DMatch> ratio_matches;
    std::vector<cv::DMatch> ratio_failures;

    // find 2 best matches for each descriptor; latter is for ratio test
    matcher.knnMatch(descriptors_cam, descriptors_map, matches, 2);

    //-- second neighbor ratio test
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < matches[i][1].distance * RATIO)
        { ratio_matches.push_back(matches[i][0]); }
        else { ratio_failures.push_back(matches[i][0]); }
    }

    double max_dist = 0; double min_dist = 1000;

    //-- compute the maximum and minimum distances between matched keypoints
    for( int i = 0; i < descriptors_cam.rows; i++ )
    {
        double dist = ratio_matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector<DMatch> good_matches = ratio_matches;
    std::vector<DMatch>  bad_matches = ratio_failures;

    //-- find matched keypoints
    //std::vector<KeyPoint> obj_keypts = keypoints_cam; // keypoints of changed object / umnatched keypoints
    std::vector<KeyPoint> matched_keypts; // keypoints of matched objects
    
    std::vector<Point2f> list_points2d_scene_match; // container for the model 2D coordinates found in the scene
    std::vector<Point3f> list_points3d_model_match; // container for the model 3D coordinates found in the scene

    for( int i = good_matches.size()-1; i >= 0; i-- )
    {
        float xf = recon.points.at(good_matches[i].trainIdx).x;
        float yf = recon.points.at(good_matches[i].trainIdx).y;
        float zf = recon.points.at(good_matches[i].trainIdx).z;
        Point3f point3d_model( xf, yf, zf ); // 3D point from model
        Point2f point2d_scene = keypoints_cam[ good_matches[i].queryIdx ].pt;  // 2D point from the scene
        list_points3d_model_match.push_back( point3d_model ); // add 3D point
        list_points2d_scene_match.push_back( point2d_scene ); // add 2D point

        matched_keypts.push_back( keypoints_cam[ good_matches[i].queryIdx ] );
        //obj_keypts.erase( obj_keypts.begin() + good_matches[i].queryIdx );
    }

    //-- debug output
    //printf("\n----Total Keypoints : %lu\n", keypoints_cam.size());
    //printf("--Matched Keypoints : %lu\n", matched_keypts.size());
    //printf("---Object Keypoints : %lu\n\n", obj_keypts.size());
    

    //-- Camera Params --//////////////////////////////////////////////////////
    // camera parameters - TODO retrieve from COLMAP
    unsigned image_size_x = 4032, focal_length_x = 4838.4, 
             image_size_y = 3024, focal_length_y = 4838.4,
             skew = 0;

    // camera intrinsics matrix
    cv::Mat cam_mat = (Mat_<double>(3,3) <<  
                               focal_length_x,           skew, image_size_x/2,
                                            0, focal_length_y, image_size_y/2,
                                            0,              0,              1);
    
    //-- RANSAC --/////////////////////////////////////////////////////////////
    cv::Mat dist_coeffs; // distortion cooefficients vector; no distortion
    
    // camera extrinsics
    cv::Mat rvec;
    cv::Mat tvec;

    cv::solvePnPRansac( list_points3d_model_match, list_points2d_scene_match, cam_mat, dist_coeffs, rvec, tvec, false, 100, 8.0, 100 );

    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);

    std::cout << " DCM: " << rot_mat << std::endl;
    std::cout << "tvec: " << tvec    << std::endl;

    //-- projection --/////////////////////////////////////////////////////////

    // IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    // 1 0.0463085 0.0926584 0.817966 0.565863 -0.743449 -3.23341 7.8009 1 frame00038.jpg
    
    // // translation vector for camera extrinsics   
    // cv::Mat tvec = (Mat_<double>(3,1) << -0.743449, -3.23341, 7.8009);

    // matrix for projected points
    cv::Mat proj_pos;

    // convert Eigen matrix of 3D point cloud to CV matrix
    cv::Mat pos_mat_cv;
    cv::eigen2cv( recon.pos_mat, pos_mat_cv );

    // project 3D point cloud to 2D camera view
    cv::projectPoints( pos_mat_cv, rvec, tvec, cam_mat, dist_coeffs, proj_pos );


    //-- directed search --////////////////////////////////////////////////////

    //-- match camera and map descriptors using FLANN matcher
    FlannBasedMatcher matcher2;

    std::vector<cv::DMatch> ratio_matches2;
    std::vector<cv::DMatch> ratio_failures2;
    std::vector<cv::KeyPoint> obj_keypts;


    // loop through each camera image point
    for ( unsigned i = 0; i < keypoints_cam.size(); i++ )
    {
        std::vector<std::vector<cv::DMatch>> matches2;

        // extract pixel coordinates
        float cam_x = keypoints_cam.at(i).pt.x;
        float cam_y = keypoints_cam.at(i).pt.y;

        std::vector<unsigned> map_region_idx;
        cv::Mat descriptors_region_cam;
        cv::Mat descriptors_region_map;

        descriptors_region_cam.push_back( descriptors_cam.row(i) );

        // check whether each projected map point is within viscinity of camera point
        for ( unsigned j = 0; j < proj_pos.rows; j++ )
        {
            float map_x = proj_pos.at<float>(j,0);
            float map_y = proj_pos.at<float>(j,1);

            float dist = sqrt( pow((cam_x - map_x),2) + pow((cam_y - map_y),2) );
            float eps = 0.05*image_size_x;

            if( dist <= eps && dist != 0.0f )
            {
                map_region_idx.push_back(j);
                descriptors_region_map.push_back( descriptors_map.row(j) );
            }
        }

        if ( !descriptors_region_map.empty() && descriptors_region_map.rows >= 2 )
        {
            // find 2 best matches for each descriptor; latter is for ratio test
            matcher2.knnMatch(descriptors_region_cam, descriptors_region_map, matches2, 2);

            //-- second neighbor ratio test
            for ( unsigned k = 0; k < matches2.size(); k++ )
            {
                if ( matches2[k][0].distance < matches2[k][1].distance * 0.9 )
                {
                    ratio_matches2.push_back( matches2[k][0] );
                }
                else
                {
                    ratio_failures2.push_back( matches2[k][0] );
                    obj_keypts.push_back( keypoints_cam.at(i) );    
                }
            }
        }
        else { obj_keypts.push_back( keypoints_cam.at(i) ); }
    }

    //-- debug output
    printf("\n----Total Keypoints : %lu\n", keypoints_cam.size());
    printf("--Matched Keypoints : %lu\n", keypoints_cam.size() - obj_keypts.size());
    printf("---Object Keypoints : %lu\n\n", obj_keypts.size());

    // stop clock
    t = clock() - t;
    printf ("Feature correlation time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);


    
    //-- clustering --/////////////////////////////////////////////////////////

    // start clock
    t = clock();

    //-- cluster remaining unmatched keypoints
    float eps = atof( config_params.at("cluster_eps").c_str() );;
    int min_pts = atoi( config_params.at("cluster_min_pts").c_str() );
    std::vector< std::vector<KeyPoint> > point_clusters = DBSCAN_keypoints( &obj_keypts, eps, min_pts );

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

    //-- clustered KeyPoints image display output
    for ( unsigned i = 0; i < point_clusters.size(); i++ )
    {

        std::vector<KeyPoint> current_cluster = point_clusters[i];
        
        int current_cluster_size = current_cluster.size();

        printf("Cluster:%d\tSize:%d\n", i, current_cluster_size);

        int min_size = atoi( config_params.at("cluster_min_size").c_str() );

        if ( current_cluster_size > min_size )
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
    printf ("Clustering time: %f seconds\n",((float)t)/CLOCKS_PER_SEC);

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
    printf(" ./main <path_to_points3D.txt> <path_to_database.db> <path_to_config.ini>\n");
    printf(" ./main <path_to_points3D.txt> <path_to_database.db> <path_to_save_serialised_data.txt> <path_to_config.ini>\n");
    printf(" ./main <path_to_load_serialised_data.txt> <path_to_config.ini>\n");

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