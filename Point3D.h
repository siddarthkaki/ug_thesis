#ifndef POINT3D_H
#define POINT3D_H

#include <stdio.h>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXi;
using Eigen::VectorXi;

class Point3D
{
    private:

    public:
        unsigned POINT3D_ID;
        double x, y, z;
        unsigned num_tracks;

        unsigned num_descriptors; // should be same as num_tracks...

        MatrixXi point_descriptors; // all descriptors for this point
        VectorXi mean_descriptor; // colwise mean of point_descriptors

        // vectors of TRACK elements
        std::vector<unsigned> IMAGE_ID;
        std::vector<unsigned> POINT2D_IDX;

        Point3D();
        Point3D( unsigned id_arg,
                 double x_arg, double y_arg, double z_arg,
                 unsigned num_tracks_arg );
};

#endif