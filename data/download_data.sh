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
tar xjvf -C ..
cd ..
rm -r temp/*
