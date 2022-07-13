#!/bin/bash

mkdir data/

curl https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip -o data/facial_keypoints.zip

unzip data/facial_keypoints.zip -d data

rm data/facial_keypoints.zip
