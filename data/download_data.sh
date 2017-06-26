#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd $SCRIPTPATH
mkdir temp
cd temp
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
unzip vehicles.zip -d ..
rm vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
unzip non-vehicles.zip -d ..
wget http://bit.ly/udacity-annotations-autti
mv udacity-annotations-autti udacity-annotations-autti.tar.gz
tar -C .. -xzf udacity-annotations-autti.tar.gz
cd ..
rm -r temp/*
