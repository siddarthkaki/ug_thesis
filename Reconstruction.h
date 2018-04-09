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
#include <boost/serialization/vector.hpp>

#include "Point3D.h"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Matrix3d;
using Eigen::Vector3d;

class Reconstruction 
{ 
    private:
        // allow serialization to access non-public data members
        friend class boost::serialization::access;

        template<typename Archive>
        void serialize(Archive& ar, const unsigned version)
        {
            ar & points_file_id;
            ar & db_file_id;
            ar & num_points;

            ar & pos_mat;
            //ar & mean_descriptors;
            //ar & all_descriptors;
            ar & point_descriptors;

            ar & points;
        }

    public:
        std::string points_file_id;
        std::string db_file_id;
        unsigned num_points;

        MatrixXd pos_mat; // matrix of 3D position for each point
        MatrixXi point_descriptors;
        //MatrixXi mean_descriptors; // matrix of mean 128-length SIFT descriptors for each point
        //MatrixXi all_descriptors; // matrix of all 128-length SIFT descriptors for each point

        std::vector<Point3D> points;

        Reconstruction();
        Reconstruction( std::string points_id_arg, 
                        std::string db_id_arg );

        unsigned LineCount( std::string filename );

        void Load();
        Reconstruction Load(std::string file_name);
        void Save(std::string file_name, Reconstruction recon_obj);
        unsigned DescriptorCount(sqlite3 &db);
        MatrixXd CamProjection(MatrixXd KMat, Matrix3d RCI, Vector3d tVec);

};
/*
namespace boost
{
    template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void serialize
    (
        Archive & ar, 
        Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
        const unsigned int file_version
    ) 
    {
        size_t rows = t.rows(), cols = t.cols();
        ar & rows;
        ar & cols;
        if( rows * cols != t.size() )
        t.resize( rows, cols );

        for(size_t i=0; i<t.size(); i++)
        ar & t.data()[i];
    }
}
*/

#endif