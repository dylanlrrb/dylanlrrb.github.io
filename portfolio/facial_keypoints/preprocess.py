import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import random
import cv2

def grayscale_resize_rescale(img, size):
  # if image has an alpha color channel, remove it
  if(img.shape[2] == 4):
      img = img[:,:,0:3]
  img = cv2.resize(img, size)
  img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
  # normalize pixel values between 0 and 1
  img = img / 255.0
  return img


def dataGenerator(annotations_path, images_path, training=False, input_size=(224,224), batch_size=8, shuffle=True):
  annotations = pd.read_csv(annotations_path).to_numpy()
  while True:
    if shuffle:
      np.random.shuffle(annotations)

    X_batch = []
    y_batch = []

    for annotation in annotations:
      image_filename = annotation[0]
      keypoints = annotation[1:].astype('float').reshape(-1, 2)

      X = np.asarray(Image.open(f'{images_path}/{image_filename}'))
      height, width = X.shape[:2]

      # if image has an alpha color channel, remove it
      if(X.shape[2] == 4):
          X = X[:,:,0:3]

      if training:
        X = color_jitter(X)
      
      X = cv2.resize(X, input_size)
      X = np.array(cv2.cvtColor(X, cv2.COLOR_RGB2GRAY))
      # normalize pixel values between 0 and 1
      X = X / 255.0

      # normalize keypoint coordinates between -1 and 1
      y = (keypoints / [width/2, height/2] - 1).flatten()

      if training:
        X, y = random_center_crop(X, y, (1., 0.5))
        X, y = random_horizontal_flip(X, y, percent=20)

      X_batch.append(X)
      y_batch.append(y)

      if len(X_batch) == batch_size:
        yield np.array(X_batch), np.array(y_batch)
        X_batch = []
        y_batch = []

def color_jitter(X, brightness_range=(-50,50), saturate_range=(-50,50)):
  brighten = int(random.randrange(*brightness_range))
  saturate = int(random.randrange(*saturate_range))
  hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(hsv)
  v = np.array(v) + brighten
  v = v.clip(0, 255).astype(h.dtype)
  s = np.array(s) + saturate
  s = s.clip(0, 255).astype(h.dtype)
  final_hsv =  cv2.merge((h, s, v))
  return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

def random_center_crop(X, y, scale_range):
  original_dim = X.shape[0]
  scale_factor = random.uniform(*scale_range)
  new_dim = int(original_dim * scale_factor)
  offset = int((original_dim - new_dim) / 2)
  cropped_X = X[offset:offset+new_dim, offset:offset+new_dim]
  cropped_X = np.array(cv2.resize(cropped_X, (original_dim, original_dim)))
  cropped_y = (y / (scale_factor)).clip(min=-0.99, max=0.99)
  return cropped_X, cropped_y

def random_horizontal_flip(X, y, percent=50):
  flipped_X = X
  flipped_y = y
  if random.randrange(0, 100) > percent:
    flipped_X = np.array(cv2.flip(X, 1))
    flipped_y = (y.reshape(-1, 2) * [-1 ,1]).flatten()
  return flipped_X, flipped_y

  
def to_tf_dataset(data_gen, batch_size=8, img_size=(224,224), output_size=136):

  def gen():
    for X_batch, y_batch in data_gen:
      yield X_batch, y_batch

  return tf.data.Dataset.from_generator(
     gen,
     output_signature=(
        tf.TensorSpec(shape=(batch_size,*img_size), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,output_size), dtype=tf.float32))
    ).prefetch(1)