/**
 * @file Object localiser
 * @brief 
 * @author S. Kaki
 */

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>


#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/nonfree.hpp"
# include "opencv2/nonfree/features2d.hpp"
# include "opencv2/calib3d/calib3d.hpp"

using namespace cv;


void readme();
std::map<std::string,std::string> load_config(std::string filename);
Rect bounding_box(Mat *img_cam, std::vector<KeyPoint> *keypoints);
std::vector< std::vector<KeyPoint> > DBSCAN_keypoints(std::vector<KeyPoint> *keypoints, float eps, int minPts);
std::vector<int> region_query(std::vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps);


/**
 * @function main
 * @brief main function
 */
int main( int argc, char** argv )
{
    if( argc != 4 )
    { readme(); return -1; }

    // read in config file
    //std::map<std::string,std::string> config_params = load_config(argv[1]);

    // read in images
    Mat img_cam = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE ); // camera image
    Mat img_map = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE ); // "map" image

    if( !img_cam.data || !img_map.data )
    { printf(" --(!) Error reading images \n"); return -1; }

    // read in threshold for discriptor "distance" comparison
    float threshold = atof (argv[3]);

    //-- Step 1: Detect the keypoints using SURF/SIFT Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_cam, keypoints_map;

    detector.detect( img_cam, keypoints_cam );
    detector.detect( img_map, keypoints_map );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_cam, descriptors_map;

    extractor.compute( img_cam, keypoints_cam, descriptors_cam );
    extractor.compute( img_map, keypoints_map, descriptors_map );

    //printf("%d\t%d\n", descriptors_cam.rows, descriptors_cam.cols );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_cam, descriptors_map, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Compute the maximum and minimum distances between matched keypoints
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
    std::vector<KeyPoint> obj_keypts = keypoints_cam;
    std::vector<KeyPoint> matched_keypts;
    //std::vector<Point2f> map_pts;

    for( int i = good_matches.size()-1; i >= 0; i-- )
    {
        //if (i < 50){ printf("%d\n", good_matches[i].queryIdx); }
        
        // TODO fix
        matched_keypts.push_back( keypoints_cam[ good_matches[i].queryIdx ] );
        obj_keypts.erase( obj_keypts.begin() + good_matches[i].queryIdx );
        //map_pts.push_back( keypoints_map[ good_matches[i].trainIdx ].pt );
    }

    //-- cluster remaining unmatched keypoints
    float eps = 35;
    int minPts = 5;
    std::vector< std::vector<KeyPoint> > point_clusters = DBSCAN_keypoints( &obj_keypts, eps, minPts );


    //-- Debug output
    printf("\n----Total Keypoints : %lu\n", keypoints_cam.size());
    printf("--Matched Keypoints : %lu\n", matched_keypts.size());
    printf("---Object Keypoints : %lu\n\n", obj_keypts.size());


    //-- Image display output

    if(0)
    {
        Mat out_img_1, out_img_2, out_img_3;

        drawKeypoints(img_cam, obj_keypts, out_img_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "Object localisation", out_img_1 );

        drawKeypoints(img_cam, matched_keypts, out_img_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "Matched Keypoints", out_img_2 );

        drawKeypoints(img_cam, keypoints_cam, out_img_3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow( "All Keypoints", out_img_3 );
    }

    Mat img_cam_rgb = imread( argv[1], CV_LOAD_IMAGE_COLOR ); // camera image in colour

    //-- Clustered KeyPoints image display output
    for (int i = 0; i < point_clusters.size(); i++)
    {

        std::vector<KeyPoint> current_cluster = point_clusters[i];
        
        int current_cluster_size = current_cluster.size();

        printf("Cluster:%d\tSize:%d\n", i, current_cluster_size);

        if (current_cluster_size > 100)
        {

            Rect ROI = bounding_box(&img_cam, &current_cluster);

            //Mat img_cam_cropped = img_cam;
            Mat img_cam_cropped = img_cam_rgb(ROI);

            Mat out_img_temp;
            drawKeypoints( img_cam_cropped, current_cluster, out_img_temp, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
            
            std::ostringstream oss;
            oss << "Cluster:" << i;
            std::string title_text = oss.str();
            imshow( title_text, out_img_temp );
        }
    }

    waitKey(0);

    return 0;
}




/******************************** FUNCTIONS **********************************/

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./obj_localiser <img_from_camera> <img_from_map>\n"); }

/**
 * @function load_config
 * @brief loads config params from ini file
 * @return map of key-value pairs
 */
std::map<std::string,std::string> load_config(std::string filename)
{
    std::ifstream input(filename); //The input stream
    std::map<std::string,std::string> ans; //A map of key-value pairs in the file
    while(input) //Keep on going as long as the file stream is good
    {
        std::string key; //The key
        std::string value; //The value
        std::getline(input, key, ':'); //Read up to the : delimiter into key
        std::getline(input, value, '\n'); //Read up to the newline into value
        std::string::size_type pos1 = value.find_first_of("\""); //Find the first quote in the value
        std::string::size_type pos2 = value.find_last_of("\""); //Find the last quote in the value
        if(pos1 != std::string::npos && pos2 != std::string::npos && pos2 > pos1) //Check if the found positions are all valid
        {
            value = value.substr(pos1+1,pos2-pos1-1); //Take a substring of the part between the quotes
            ans[key] = value; //Store the result in the map
        }
    }
    input.close(); //Close the file stream
    return ans; //And return the result
}

/**
 * @function bounding_box
 * @brief provides bounding box corners for a set of keypoints
 */
Rect bounding_box(Mat *img_cam, std::vector<KeyPoint> *keypoints)
{
    float min_x = img_cam->cols, max_x = 0, min_y = img_cam->rows, max_y = 0;

    for (int i = 0; i < keypoints->size(); i++)
    {
        if (keypoints->at(i).pt.x < min_x) { min_x = keypoints->at(i).pt.x; }
        if (keypoints->at(i).pt.x > max_x) { max_x = keypoints->at(i).pt.x; }
        if (keypoints->at(i).pt.y < min_y) { min_y = keypoints->at(i).pt.y; }
        if (keypoints->at(i).pt.y > max_y) { max_y = keypoints->at(i).pt.y; }
        //printf("TEST\n");
    }

    //printf("%f\t%f\t%f\t%f\n", min_x, max_x, min_y, max_y);

    Rect ROI(min_x, min_y, abs(max_x - min_x), abs(max_y - min_y));

    return ROI;
}

/**
 * @function DBSCAN_keypoints
 * @brief density-based spatial clustering of applications with noise
 */
std::vector< std::vector<KeyPoint> > DBSCAN_keypoints(std::vector<KeyPoint> *keypoints, float eps, int minPts)
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
            neighbor_pts = region_query(keypoints, &keypoints->at(i), eps);
            if(neighbor_pts.size() < minPts)
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
                        neighbor_pts_ = region_query(keypoints,&keypoints->at(neighbor_pts[j]),eps);
                        if(neighbor_pts_.size() >= minPts)
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
 * @function region_query
 * @brief searching for closest keypoints to keypoint in question
 */
vector<int> region_query(std::vector<KeyPoint> *keypoints, KeyPoint *keypoint, float eps)
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

#endif