#!/bin/bash
export VLROOT=$PWD/ext_libs/vlfeat
export LD_PRELOAD=$VLROOT/bin/glnxa64/libvl.so
g++ -g main.cpp -o main Reconstruction.cpp Point3D.cpp -std=c++11 -I$VLROOT -L$VLROOT/bin/glnxa64 -lvl -lsqlite3 -lboost_serialization $(pkg-config --libs opencv --cflags)
exit 0
