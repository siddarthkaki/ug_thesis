#include "Point3D.h"

Point3D::Point3D() {}
Point3D::Point3D( unsigned id_arg,
                  double x_arg, double y_arg, double z_arg,
                  unsigned num_tracks_arg )
{
    POINT3D_ID = id_arg;
    x = x_arg;
    y = y_arg;
    z = z_arg;
    num_tracks = num_tracks_arg;

    //point_descriptors.resize(num_tracks,128);
    point_descriptor.resize(128);

    IMAGE_ID.reserve(num_tracks);
    POINT2D_IDX.reserve(num_tracks);
}