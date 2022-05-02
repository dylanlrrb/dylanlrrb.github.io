import os
import tensorflow as tf
# from tensorflow.keras import layers
from keras.layers import RandomFlip, RandomRotation, Resizing, RandomCrop, Rescaling
from dataset_helpers import split_training_image_dataset
from generator import define_generator
from perceptual_loss import PerceptualLoss

# 1 = INFO messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


batch_size = 2
image_size = 512
patch_size = 256
resolution_down_factor = 0.25

data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),])

reduce_resolution = tf.keras.Sequential([
  Resizing(int(patch_size*resolution_down_factor), int(patch_size*resolution_down_factor)),
  Resizing(patch_size, patch_size),])

rand_crop = RandomCrop(patch_size, patch_size)

tanh_scale =  Rescaling(1./127.5, offset=-1)

def train_transforms(x):
  y = rand_crop(data_augmentation(x))
  X = reduce_resolution(y)
  return (X, tanh_scale(y))

def val_transforms(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, tanh_scale(y))

def test_transforms(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, tanh_scale(y))


train_ds, val_ds, test_ds = split_training_image_dataset(
  'flickr30k_images',
  'https://datasets-349058029.s3.us-west-2.amazonaws.com/flickr/flickr30k_images.zip',
  train_transforms,
  val_transforms,
  test_transforms,
  labels=None,
  data_subsample=0.1,
  batch_size=batch_size,
  image_size=(image_size,image_size),
)


# Build Model
tf.keras.backend.clear_session()
auto_encoder = define_generator((patch_size, patch_size, 3))


# Training
model_name = 'generator_4Xzoom_plossX0-1'
epochs = 10

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
  filepath=f'models/{model_name}.h5',
  monitor='val_loss',
  mode='min',
  save_best_only=True)

auto_encoder.compile(optimizer='adam', loss=PerceptualLoss(0.1))
auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[model_checkpoint])
