#!/bin/bash
g++ main.cpp -o main Reconstruction.cpp Point3D.cpp -std=c++11 -lsqlite3 -lboost_serialization $(pkg-config --libs opencv --cflags)
exit 0