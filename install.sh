#!/bin/sh

if [ ! -x "./build" ]; then
	mkdir build
fi

cd build
cmake ..
make 
cd ..
