import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

class CategoryMappingHelper():
  def __init__(self, coco_annotations):
    self.catIds = coco_annotations.getCatIds()
    self.categories = [*coco_annotations.loadCats(self.catIds), {'id': 0, 'name': 'background'}]
    self.filter_idx_to_category_id = [cat['id'] for cat in self.categories]
    self.filter_idx_to_category_name = {idx: cat['name'] for idx, cat in enumerate(self.categories)}
    self.categry_id_to_category_name = {cat['id']: cat['name'] for cat in self.categories}
    self.category_id_to_filter_idx = {id: idx for idx, id in enumerate(self.filter_idx_to_category_id)}
    self.category_name_to_filter_idx = {name: idx for idx, name in enumerate(self.filter_idx_to_category_name)}
  def get (self, value, in_key, out_key):
    if   (in_key == 'filter_idx' and out_key == 'category_id'):
      return self.filter_idx_to_category_id[value]
    elif (in_key == 'filter_idx' and out_key == 'category_name'):
      return self.filter_idx_to_category_name[value]
    elif (in_key == 'category_id' and out_key == 'category_name'):
      return self.categry_id_to_category_name[value]
    elif (in_key == 'category_id' and out_key == 'filter_idx'):
      return self.category_id_to_filter_idx[value]
    elif (in_key == 'category_name' and out_key == 'filter_idx'):
      return self.category_name_to_filter_idx[value]
    else:
      raise Exception("in_key:out_key pair not supported. Valid pairs: \nfilter_idx:category_id\nfilter_idx:category_name\ncategory_id:category_name\ncategory_id:filter_idx\ncategory_name:filter_idx")


def cocoDataGenerator(coco_annotations, image_folder, catMapper, batch_size, shuffle=False, input_size=(128,128), output_size=(128,128), train=False):
  while True:
    numCategories = len(catMapper.catIds)
    imageIds = coco_annotations.getImgIds()
    if shuffle:
      np.random.shuffle(imageIds)

    if train:
      imageIds=[]
      catIDs = coco_annotations.getCatIds()
      cats = coco_annotations.loadCats(catIDs)
      class_freq = [len(coco_annotations.getImgIds(catIds=cat['id'])) for cat in cats]
      max_freq = np.max(class_freq)

      for id in catIDs:
        imgIds = coco_annotations.getImgIds(catIds=[id])
        numImgs = len(imgIds)
        repeat_num = max_freq//numImgs
        imgIds = (np.array(imgIds * (repeat_num + 1)).flatten())[:max_freq]
        imageIds.append(imgIds) 
      
      imageIds = (np.array(imageIds)).flatten().tolist()
      np.random.shuffle(imageIds)

      
    X_batch = []
    y_batch = []
    
    for imageId in imageIds:
      img_meta = coco_annotations.loadImgs(imageId)[0]
      ann_ids = coco_annotations.getAnnIds(imgIds=[imageId], iscrowd=None)
      anns = coco_annotations.loadAnns(ann_ids)

      X = Image.open(f'{image_folder}/{img_meta["file_name"]}')
      X = np.array(X)
      X = cv2.resize(X, input_size)

      masks = [np.zeros(output_size) for _ in range(numCategories)]
      bg = np.zeros(output_size)
      for ann in anns:
        new_mask = cv2.resize(coco_annotations.annToMask(ann), output_size)
        # create mask for the background
        bg = np.maximum(new_mask, bg)
        prev_mask = masks[catMapper.get(ann['category_id'], 'category_id', 'filter_idx')]
        combined_masks = np.maximum(new_mask, prev_mask)
        masks[catMapper.get(ann['category_id'], 'category_id', 'filter_idx')] = combined_masks
      bg =  np.logical_not(bg).astype(int)
      masks.append(bg)
      y = np.dstack(masks)
      
      if X.shape == (*input_size,3) and y.shape == (*output_size,numCategories+1):
        X_batch.append(X)
        y_batch.append(y)

      if len(X_batch) == batch_size:
        yield np.array(X_batch), np.array(y_batch)

        X_batch = []
        y_batch = []



def to_tf_dataset(data_gen, batch_size, img_size, num_classes):

  def gen():
    for X_batch, y_batch in data_gen:
      yield X_batch, y_batch

  return tf.data.Dataset.from_generator(
     gen,
     output_signature=(
        tf.TensorSpec(shape=(batch_size,*img_size,3), dtype=tf.int32),
        tf.TensorSpec(shape=(batch_size,*img_size,num_classes), dtype=tf.float32))
    ).prefetch(1)
    