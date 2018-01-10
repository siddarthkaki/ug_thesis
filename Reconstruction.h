#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

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
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "Point3D.h"

using Eigen::MatrixXd;
using Eigen::MatrixXi;

class Reconstruction 
{ 
    private:
        /*
        friend class boost::serialization::access;
        // When the class Archive corresponds to an output archive, the
        // & operator is defined similar to <<.  Likewise, when the class Archive
        // is a type of input archive the & operator is defined similar to >>.
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & points_file_id;
            ar & db_file_id;
            ar & num_points;

            ar & pos_mat;
            ar & mean_descriptors;

            ar & points;
        }
        */
    public:
        std::string points_file_id;
        std::string db_file_id;
        unsigned num_points;

        MatrixXd pos_mat; // matrix of 3D position for each point
        MatrixXi mean_descriptors; // matrix of mean 128-length SIFT descriptors for each point

        std::vector<Point3D> points;

        Reconstruction();
        Reconstruction( std::string points_id_arg, 
                        std::string db_id_arg );

        unsigned LineCount( std::string filename );

        void Load();
        // Reconstruction Load(std::string file_name);
        // void Save(Reconstruction recon_obj, std::string file_name);

};

#endif