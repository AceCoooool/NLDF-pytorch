#!/usr/bin/env bash
# ECSSD images
URL=http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip
ZIP_FILE=./data/images.zip
mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE

# ECSSD labels
URL=http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip
ZIP_FILE=./data/ground_truth_mask.zip
mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE