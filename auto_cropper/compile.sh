#!/bin/bash
g++ obj_localiser.cpp -o obj_localiser -std=c++11 $(pkg-config --libs opencv --cflags)
exit 0