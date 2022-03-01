#!/bin/bash

curl https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip -o landmark_images.zip

unzip landmark_images.zip

rm landmark_images.zip
