#!/bin/bash

echo "****************** Installing pytorch ******************"
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python


echo ""
echo ""
echo "****************** cupy (needed for PWCNet used in real-world track) ******************"
conda install -c conda-forge cupy=7.8.0 

echo ""
echo ""
echo "****************** exifread ******************"
pip install exifread

echo ""
echo ""
echo "****************** Installation complete! ******************"
