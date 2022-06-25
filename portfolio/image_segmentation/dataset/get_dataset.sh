#!/bin/bash

# ~/datasets
# └───coco2017  
#     └───images
#     │   └───train2017
#     │   │    │   000000000009.jpg
#     │   │    │   000000000025.jpg
#     │   │    │   ...
#     │   └───val2017  
#     │        │   000000000139.jpg
#     │        │   000000000285.jpg
#     │        │   ...
#     └───annotations
#         │   instances_train2017.json
#         │   instances_val2017.json
#         │   captions_train2017.json
#         │   captions_val2017.json
#         │   person_keypoints_train2017.json
#         │   person_keypoints_val2017.json

curl --create-dirs --output ~/datasets/coco2017/images/val2017.zip "http://images.cocodataset.org/zips/val2017.zip"

unzip ~/datasets/coco2017/images/val2017.zip -d ~/datasets/coco2017/images/

rm ~/datasets/coco2017/images/val2017.zip




curl --create-dirs --output ~/datasets/coco2017/images/train2017.zip "http://images.cocodataset.org/zips/train2017.zip"

unzip ~/datasets/coco2017/images/train2017.zip -d ~/datasets/coco2017/images/

rm ~/datasets/coco2017/images/train2017.zip




curl --create-dirs --output ~/datasets/coco2017/stuff_annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

unzip ~/datasets/coco2017/stuff_annotations_trainval2017.zip -d ~/datasets/coco2017/

rm ~/datasets/coco2017/stuff_annotations_trainval2017.zip
