#!/bin/bash
g++ main.cpp -o main -std=c++11 -lsqlite3 $(pkg-config --libs opencv --cflags)
exit 0
