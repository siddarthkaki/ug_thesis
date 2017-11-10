#!/bin/bash
g++ obj_localiser.cpp -o obj_localiser $(pkg-config --libs opencv --cflags)
exit 0