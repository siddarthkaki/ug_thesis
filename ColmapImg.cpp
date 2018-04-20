#include "ColmapImg.h"

using Eigen::MatrixXi;
using Eigen::VectorXi;

ColmapImg::ColmapImg(std::map<std::string,std::string> arg_config_params)
{
    config_params = arg_config_params;
    cm_database_path = ".temp.db";
}

/**
 * @function ColmapSiftFeatures
 * @brief COLMAP implementation of SIFT feature extraction
 */
void ColmapImg::ColmapSiftFeatures()
{
    // COLMAP SYS CALL
    std::string cm_image_path = config_params.at("img_cam");
    system("mkdir .temp_img");
    std::string sys_call = "cp " + cm_image_path + " .temp_img/";
    system(sys_call.c_str());
    //std::cout << cm_database_path << std::endl;
    std::string cm_sys_call = "colmap feature_extractor --database_path " + cm_database_path + " --image_path .temp_img --SiftExtraction.max_num_features 2000 --SiftExtraction.use_gpu 0";
    std::cout << cm_sys_call << std::endl;
    int cm_sys_res = system(cm_sys_call.c_str());
}

/**
 * @function ColmapSiftKeypoints
 * @brief COLMAP implementation of SIFT keypoints
 */
std::vector<cv::KeyPoint> ColmapImg::ColmapSiftKeypoints()
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
cv::Mat ColmapImg::ColmapSiftDescriptors()
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

    cv::eigen2cv(img_descriptors, descriptors_cam);

    return descriptors_cam;
}

/**
 * @function ColmapClean
 * @brief Clean up disk files of COLMAP processing
 */
void ColmapImg::ColmapClean()
{
    system("rm -rf .temp_img");

    std::string sys_call = "rm " + cm_database_path + "*";
    system(sys_call.c_str());
}