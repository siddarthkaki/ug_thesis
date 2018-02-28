#ifndef POINT3D_H
#define POINT3D_H

#include <stdio.h>
#include <eigen3/Eigen/Dense>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

using Eigen::MatrixXi;
using Eigen::VectorXi;

// forward declaration of class boost::serialization::access
namespace boost { namespace serialization { class access; } }

class Point3D
{
    private:
        // allow serialization to access non-public data members
        friend class boost::serialization::access;

        template<typename Archive>
        void serialize(Archive& ar, const unsigned version)
        {
            ar & POINT3D_ID;
            ar & x & y & z;
            ar & num_tracks;

            ar & num_descriptors;

            ar & point_descriptor;
            //ar & mean_descriptor;

            ar & IMAGE_ID;
            ar & POINT2D_IDX;
        }

    public:
        unsigned POINT3D_ID;
        double x, y, z;
        unsigned num_tracks;

        unsigned num_descriptors; // should be same as num_tracks...

        //MatrixXi point_descriptors; // all descriptors for this point
        VectorXi point_descriptor; // point descriptor

        // vectors of TRACK elements
        std::vector<unsigned> IMAGE_ID;
        std::vector<unsigned> POINT2D_IDX;

        Point3D();
        Point3D( unsigned id_arg,
                 double x_arg, double y_arg, double z_arg,
                 unsigned num_tracks_arg );
};

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

#endif