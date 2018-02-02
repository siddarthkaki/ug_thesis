#include "Reconstruction.h" 

Reconstruction::Reconstruction() {}
Reconstruction::Reconstruction( std::string points_id_arg, 
                                std::string db_id_arg )
{
    // read in points3D.txt file name
    points_file_id = points_id_arg;

    // read in database file name
    db_file_id = db_id_arg;

    // number of 3D points in file
    num_points = LineCount(points_file_id) - 3;

    printf("Num 3D points: %u\n", num_points);

    pos_mat.resize(num_points,3); // matrix of 3D position for each point
    mean_descriptors.resize(num_points,128); // matrix of mean 128-length SIFT descriptors for each point   
}

void Reconstruction::Load()
{
    // setup db connection
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;
    rc = sqlite3_open(db_file_id.c_str(), &db);
    if( rc ) { fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db)); return; }
    else     { fprintf(stderr, "Opened database successfully\n"); }
    //sqlite3_close(db);

    unsigned char *pzBlob; // holder pointer to blob
    int *pnBlob; // retrieved blob size

    std::ifstream input_file(points_file_id);
    input_file.seekg(std::ios::beg);
    std::string curr_line;

    if ( input_file.is_open() )
    {
        unsigned ind_total_tracks = 0;

        // allocate space for entire descriptors matrix
        all_descriptors.resize(Reconstruction::DescriptorCount(*db),128);

        // skip header comments
        for( unsigned i = 0; i < 3; i++ ) { getline( input_file, curr_line ); }

        // loop through each point in 3D point cloud
        for( unsigned i = 0; i < num_points; i++ )
        {
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

            unsigned num_tracks = (tokenised_double.size() - 8)/2; // number of images with this point

            // create new Point3D object
            Point3D point_temp( tokenised_double.at(0),
                                tokenised_double.at(1),
                                tokenised_double.at(2),
                                tokenised_double.at(3),
                                num_tracks);

            //MatrixXi curr_descriptors(num_tracks,128); // matrix of descriptors for each point

            // loop through each image with this point
            for( unsigned j = 0; j < num_tracks; j++ )
            {
                point_temp.IMAGE_ID.push_back(tokenised_double.at(6+2*(j+1)));
                point_temp.POINT2D_IDX.push_back(tokenised_double.at(7+2*(j+1)));

                //if (i % 9213 == 0) { std::cout << "Current IMAGE_ID: " << curr_image_id << std::endl; }
                //if (i % 9213 == 0) { std::cout << "Current POINT2D_IDX: " << curr_point2d_idx << std::endl; }

                sqlite3_stmt *stmt;
                std::string zSql = "SELECT * FROM descriptors WHERE image_id=" + std::to_string(point_temp.IMAGE_ID.at(j));
                sqlite3_prepare_v2(db, zSql.c_str(), -1, &stmt, NULL);
                rc = sqlite3_step(stmt);
                
                if (rc == SQLITE_ROW)
                {
                    unsigned curr_num_descriptors = sqlite3_column_int(stmt,1); // get num of descriptors from column 1

                    // get uint8_t BLOB data from db
                    std::vector<char> data( sqlite3_column_bytes(stmt, 3) );
                    const char *pBuffer = reinterpret_cast<const char*>( sqlite3_column_blob(stmt, 3) );
                    std::copy( pBuffer, pBuffer + data.size(), &data[0] );

                    unsigned curr_start = point_temp.POINT2D_IDX.at(j)*128; // checked
                    unsigned curr_end = curr_start+127; // checked

                    VectorXi curr_descriptor(128);

                    for ( unsigned k = 0; k < 128; k++ )
                    { curr_descriptor(k) = ( (int) ((uint8_t) data.at(curr_start + k)) ); }

                    point_temp.point_descriptors.block<1,128>(j,0) = curr_descriptor;
                    all_descriptors.block<1,128>(ind_total_tracks,0) = curr_descriptor;
                }

                rc = sqlite3_finalize(stmt);

                ind_total_tracks++;

            }

            point_temp.mean_descriptor = point_temp.point_descriptors.colwise().mean();
            mean_descriptors.block<1,128>(i,0) = point_temp.mean_descriptor;
            //if (i % 1000 == 0) { std::cout << mean_descriptor.transpose() << std::endl; }

            points.push_back(point_temp);

            if (i % 5000 == 0) { std::cout << "Completed point: " << i << std::endl; }
        }

        //std::cout << mean_descriptors << std::endl;
    }

    rc = sqlite3_close(db);
}

Reconstruction Reconstruction::Load(std::string file_name)
{
    Reconstruction new_obj;
    std::ifstream ifs(file_name);
    boost::archive::text_iarchive ia(ifs);
    ia & new_obj;
    return new_obj;
}

void Reconstruction::Save(std::string file_name, Reconstruction recon_obj)
{
    std::ofstream ofs(file_name); // open file stream
    boost::archive::text_oarchive oa(ofs);
    oa & recon_obj;
}

unsigned Reconstruction::LineCount(std::string filename)
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

unsigned Reconstruction::DescriptorCount(sqlite3 &db)
{
    unsigned count = 0;

    sqlite3_stmt *stm;
    std::string zSql = "SELECT * FROM descriptors";
    sqlite3_prepare_v2(&db, zSql.c_str(), -1, &stm, NULL);

    while (sqlite3_step(stm) != SQLITE_DONE)
    {
        count = count + sqlite3_column_int(stm,1);
    }
    
    return count;
}