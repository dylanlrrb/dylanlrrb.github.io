from cv2 import KeyPoint
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
import wandb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import gc
from PIL import Image
from preprocess import grayscale_resize_rescale
import cv2
import os

def model_checkpoint_callback(project_dir, model_name): 
  return ModelCheckpoint(
        filepath=f'{project_dir}/models/{model_name}.h5',
        monitor='loss',
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
            {'name': 'face4', 'extension': 'jpeg', 'description': 'Face 4'},]


class CustomCallback(Callback):
  def __init__(self, project_dir, model_name, image_size):
    self.model_name = model_name
    self.project_dir = project_dir
    self.image_size = image_size
    super().__init__()
  def on_epoch_end(self, epoch, logs=None):
    for sample in samples:
      image = np.asarray(Image.open(f'{self.project_dir}/assets/{sample["name"]}.{sample["extension"]}'))
      plot = create_plot(self.model, image, self.image_size, epoch)
      img = plot_to_img(plot)
      wandb.log({sample['description']: wandb.Image(img)})
    gc.collect()

class GarbageCollection(Callback):
  def __init__(self):
    super().__init__()
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()


def create_plot(model, image, image_size, epoch):
  h, w = image.shape[:2]
  in_image = np.array([grayscale_resize_rescale(image, size=image_size)])
  key_points = np.squeeze(model.predict(in_image)).reshape(-1,2)
  key_points = (key_points + 1) * [w/2, h/2]
  # key_points = np.squeeze(model.predict(in_image))
  # key_points = ((key_points + 1) * 112).reshape(-1, 2)

  figure, ax = plt.subplots()
  ax.text(0., 1., f'{epoch}'.zfill(2),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', edgecolor='none'))
  ax.imshow(image)
  # ax.imshow(grayscale_resize_rescale(image, size=image_size))
  ax.scatter(key_points[:, 0], key_points[:, 1], s=20, marker='o', c='m')
  # ax.scatter(key_points[:, 0], key_points[:, 1], s=20, marker='.', c='w')
  ax.axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  return figure


def plot_to_img(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  return image