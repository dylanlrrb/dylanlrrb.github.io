import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from PIL import Image
import random
import cv2
from scipy import ndimage


def dataGenerator(annotations_path, images_path, face_detector_path, training=False, input_size=(224,224), batch_size=8, shuffle=True):
  annotations = pd.read_csv(annotations_path).to_numpy()
  face_cascade = cv2.CascadeClassifier(face_detector_path)
  while True:
    if shuffle:
      np.random.shuffle(annotations)

    X_batch = []
    y_batch = []

    for annotation in annotations:
      image_filename = annotation[0]
      keypoints = annotation[1:].astype('float').reshape(-1, 2)

      X = Image.open(f'{images_path}/{image_filename}')
      X = np.array(X)
      # if image has an alpha color channel, remove it
      if(X.shape[2] == 4):
        X = X[:,:,0:3]

      height, width,  = X.shape[:2]
      longest_side = max(X.shape[0], X.shape[1])
      X = cv2.resize(X, (longest_side, longest_side))
      y = (keypoints * [longest_side/width, longest_side/height]).flatten()
      face_detected, X, y = crop_face(face_cascade, X, y)
      if not face_detected:
        continue
      
      height, width = X.shape[:2]
      # normalize keypoint coordinates between -1 and 1
      y = (y.reshape(-1, 2) / [width/2, height/2] - 1).flatten()

      if training:
        X, y = random_rotation(X, y, rotation_range=(-30, 30))
        # X = random_blur(X, percent=5)
        X = color_jitter(X, brightness_range=(-5,5), saturate_range=(-5,5))

      y = np.clip(y, -1., 1.)

      X = cv2.resize(X, input_size)

      X_batch.append(X)
      y_batch.append(y)

      if len(X_batch) == batch_size:
        yield np.array(X_batch), np.array(y_batch)
        X_batch = []
        y_batch = []

def color_jitter(X, brightness_range=(-10,10), saturate_range=(-10,10)):
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

def crop_face(face_cascade, X, keypoints, expand_factor=1.3):
  orig_height, orig_width  = X.shape[:2]
  gray = np.array(cv2.cvtColor(np.copy(X), cv2.COLOR_RGB2GRAY))
  # Detect the faces in image
  faces = face_cascade.detectMultiScale(gray, 1.25, 6)
  if len(faces) == 0:
    return False, None, None
  
  x,y,w,h = faces[0]
  expand_pixels = int(w * expand_factor) - w
  x,y = (max(x-int(expand_pixels/2), 0), max(y-int(expand_pixels/2), 0))
  w = (orig_width - x) if w + expand_pixels + x > orig_width else (w + expand_pixels)
  h = (orig_height - y) if h + expand_pixels + y > orig_height else (h + expand_pixels)
  X = X[y:y+h,x:x+w,:]
  cropped_keypoints = keypoints.reshape(-1, 2)
  cropped_keypoints = cropped_keypoints * [w/orig_width, h/orig_height]
  cropped_keypoints = cropped_keypoints * [(orig_width/w), orig_height/h]
  cropped_keypoints = cropped_keypoints - [x, y]
  cropped_keypoints = cropped_keypoints.flatten()
  
  return True, X, cropped_keypoints

def random_blur(X, percent=30):
  if random.randrange(0, 100) < percent:
    X = ndimage.uniform_filter(X, (2,2,2))
  return X

def random_rotation(X, y, rotation_range=(-20, 20)):
  degrees = int(random.randrange(*rotation_range))
  theta = np.radians(degrees)
  c, s = np.cos(theta), np.sin(theta)
  R = np.array(((c, -s), (s, c)))
  rotated_keypoints = np.dot(y.reshape(-1, 2), R.T)
  rotated_image = ndimage.rotate(X, degrees * -1, reshape=False)

  return rotated_image, rotated_keypoints.flatten()

def random_horizontal_flip(X, y, percent=20):
  flipped_X = X
  flipped_y = y
  if random.randrange(0, 100) < percent:
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
        tf.TensorSpec(shape=(batch_size,*img_size,3), dtype=tf.uint8),
        tf.TensorSpec(shape=(batch_size,output_size), dtype=tf.float32))
    ).prefetch(1)
