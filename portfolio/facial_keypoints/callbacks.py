from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import io
import gc
from PIL import Image
import cv2
import os

def model_checkpoint_callback(project_dir, model_name): 
  return ModelCheckpoint(
        filepath=f'{project_dir}/models/{model_name}.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True)


def reduce_lr(lr_decay_factor, lr_decay_patience, lr_decay_cooldown):
  return ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=lr_decay_factor,   
    patience=lr_decay_patience, 
    cooldown=lr_decay_cooldown,
    min_lr=0,)


samples = [ {'name': 'face1', 'extension': 'jpeg', 'description': 'Face 1'},
            {'name': 'face2', 'extension': 'jpeg', 'description': 'Face 2'},
            {'name': 'face3', 'extension': 'jpeg', 'description': 'Face 3'},
            {'name': 'face4', 'extension': 'jpeg', 'description': 'Face 4'},
            {'name': 'face5', 'extension': 'jpeg', 'description': 'Face 5'},
            {'name': 'face6', 'extension': 'jpeg', 'description': 'Face 6'},]


class CustomCallback(Callback):
  def __init__(self, project_dir, model_name, image_size, face_detector_path):
    self.model_name = model_name
    self.project_dir = project_dir
    self.image_size = image_size
    self.face_cascade = cv2.CascadeClassifier(face_detector_path)
    super().__init__()
  def on_epoch_end(self, epoch, logs=None):
    for sample in samples:
      image = np.asarray(Image.open(f'{self.project_dir}/assets/{sample["name"]}.{sample["extension"]}'))
      plot = create_plot(self.model, image, self.image_size, epoch, self.face_cascade)
      img = plot_to_img(plot)
      wandb.log({sample['description']: wandb.Image(img)})
    gc.collect()

class GarbageCollection(Callback):
  def __init__(self):
    super().__init__()
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()


def create_plot(model, image, image_size, epoch, face_cascade):
  orig_image = np.copy(image)
  orig_height, orig_width = orig_image.shape[:2]
  if(image.shape[2] == 4):
    image = image[:,:,0:3]
  longest_side = max(image.shape[0], image.shape[1])
  image = cv2.resize(image, (longest_side, longest_side))
  height_scale = orig_height / longest_side
  width_scale = orig_width / longest_side

  figure, ax = plt.subplots()
  ax.text(0., 1., f'{epoch}'.zfill(2),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', edgecolor='none'))
  # ax.imshow(image)
  ax.imshow(orig_image) # test
  ax.axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  
  # image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  gray = np.array(gray)
  # find faces
  # Detect the faces in image
  faces = face_cascade.detectMultiScale(gray, 1.25, 6)
  if len(faces) == 0:
    return figure
  
  
  face_crops = []
  # for each face
  def expand_face_detected(expand_factor=1.3):
    def lmbda(face):
      x,y,w,h = face
      expand_pixels = int(w * expand_factor) - w
      x,y = (max(x-int(expand_pixels/2), 0), max(y-int(expand_pixels/2), 0))
      w = (orig_width - x) if w + expand_pixels + x > orig_width else (w + expand_pixels)
      h = (orig_height - y) if h + expand_pixels + y > orig_height else (h + expand_pixels)
      return [x,y,w,h]
    return lmbda

  faces = [*map(expand_face_detected(), faces)]
  for x,y,w,h in faces:
    ax.add_patch(patches.Rectangle(
      xy=(x * width_scale, y * height_scale),  # point of origin.
      width=(w * width_scale), height=(h * height_scale), linewidth=1,
      color='red', fill=False))
  #   get crop of face
    crop = image[y:y+h,x:x+w,:]
  #   rescale face crop
    # crop = crop / 255.
  #   resize face crop
    crop = np.array(cv2.resize(crop, image_size))
  #  push into array
    face_crops.append(crop)
  # predict on array
  predictions = np.array(model.predict(np.array(face_crops)))
  # for each prediction
  for keypoints, face in zip(predictions, faces):
    x,y,w,h = face
    keypoints = keypoints.reshape(-1, 2)
  # calculate keypoint positions on original image
    keypoints = keypoints + 1
    keypoints = keypoints * [w/2, h/2]
    keypoints = keypoints + [x,y]
    keypoints = keypoints * [width_scale, height_scale] # test
  #   draw keypoints on original image
    ax.scatter(keypoints[:, 0], keypoints[:, 1], s=20, marker='.', c='m')
  return figure


def plot_to_img(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  return image