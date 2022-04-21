from enum import auto
from gc import callbacks
from tabnanny import verbose
from black import out
from matplotlib import image
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import sys
import pathlib
import os
from functools import reduce

hyperparameter_tuning = True
if len(sys.argv) > 1:
  if sys.argv[1] == 'full_train':
    hyperparameter_tuning = False

dataset_name = 'flickr30k_images'
data_dir = f'/root/.keras/datasets/{dataset_name}'

if not os.path.isdir(data_dir):
  dataset_url = f'https://datasets-349058029.s3.us-west-2.amazonaws.com/flickr/{dataset_name}.zip'
  tf.keras.utils.get_file(origin=dataset_url, extract=True)

data_dir = pathlib.Path(data_dir)
print(f"{len(list(data_dir.glob('*/*.jpg')))} images in dataset")

batch_size = 2
image_size = 672
patch_size = 224
resolution_down_factor = 0.2

ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels=None,
  shuffle=True,
  image_size=(image_size, image_size),
  batch_size=batch_size)

print(f'dataset size: {len(ds)}')

if hyperparameter_tuning:
  ds = ds.take(int(0.1 * len(ds)))
  print(f'dataset size used for hyperparameter tuning: {len(ds)}')

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),])

reduce_resolution = tf.keras.Sequential([
  layers.Resizing(int(patch_size*resolution_down_factor), int(patch_size*resolution_down_factor)),
  layers.Resizing(patch_size, patch_size),])

scale = tf.keras.Sequential([
  layers.RandomCrop(patch_size, patch_size),
  layers.Rescaling(1./255),])

def train_augmentation(x):
  y = scale(data_augmentation(x))
  X = reduce_resolution(y)
  return (X, y)

def val_transform(x):
  y = scale(x)
  X = reduce_resolution(y)
  return (X, y)

def test_transform(x):
  y = scale(x)
  X = reduce_resolution(y)
  return (X, y)

def shard_dataset_splits(ds, train_ratio=8, valid_ratio=1, test_ratio=1):
  num_shards = train_ratio + valid_ratio + test_ratio
  shards = [ds.shard(num_shards=num_shards, index=i) for i in range(num_shards)]
  
  train_shards = shards[:train_ratio]
  valid_shards = shards[train_ratio:train_ratio+valid_ratio]
  test_shards  = shards[train_ratio+valid_ratio:]

  train_ds = reduce(lambda a, b: a.concatenate(b), train_shards)
  val_ds = reduce(lambda a, b: a.concatenate(b), valid_shards)
  test_ds  = reduce(lambda a, b: a.concatenate(b), test_shards)

  return train_ds, val_ds, test_ds

AUTOTUNE = tf.data.AUTOTUNE

train_ds, val_ds, test_ds = shard_dataset_splits(ds)

train_ds = train_ds.map(train_augmentation, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(val_transform, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(test_transform, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)

# Build Perceptual Loss function
def perceptual_loss(prcpt_weight, layer_maps=None):
  # different number of layers
  # combine with mse at different wights
  
  feature_extractor = tf.keras.applications.MobileNetV2(
    input_shape=(patch_size, patch_size, 3),
    include_top=False,
    weights='imagenet')
  feature_extractor.trainable = False
  for layer in feature_extractor.layers:
    layer.trainable=False
  
  if layer_maps is None:
    layer_maps = ['block_2_add', 'block_4_add', 'block_8_add', 'block_12_add', 'block_15_add']

  loss_models = [tf.keras.Model(feature_extractor.inputs, feature_extractor.get_layer(m).output) for m in layer_maps]

  def loss_fn(y_true, y_pred):

    mse_loss = K.square(y_pred - y_true)
    mse_loss = K.mean(mse_loss, axis=[1,2,3])

    prcpt_loss = mse_loss

    for loss_model in loss_models:
      y_true_features = loss_model(y_true)
      y_pred_features = loss_model(y_pred)
      loss = K.square(y_pred_features - y_true_features)
      loss = K.mean(loss, axis=[1,2,3])
      prcpt_loss = prcpt_loss + loss

    return (K.mean(prcpt_loss) * prcpt_weight) + mse_loss

  return loss_fn


# Build Model
model_name = sys.argv[0].replace('train_','').replace('.py','') + ('' if hyperparameter_tuning else '_full_train')

tf.keras.backend.clear_session()

input_img = layers.Input(shape=(patch_size, patch_size, 3))

l1 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
l2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
l3 = layers.MaxPool2D(padding='same')(l2)

l4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
l5 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
l6 = layers.MaxPool2D(padding='same')(l5)

l7 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)
l8 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l7)
l9 = layers.MaxPool2D(padding='same')(l8)

l10 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)
l11 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l10)

l12 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l11)
l13 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
l14 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)

l15 = layers.add([l14, l8])

l16 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)
l17 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l16)
l18 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l17)

l19 = layers.add([l18, l5])

l20 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l19)
l21 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l20)
l22 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l21)

l23 = layers.add([l22, l2])

decoded_image = layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l23)

auto_encoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)

# Training
print('training with no perceptual loss')
auto_encoder.compile(optimizer='adam', loss=perceptual_loss(0))
auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1,
  verbose=1,)
auto_encoder.save(f'models/{model_name}_0.h5')

print('training with 0.01 perceptual loss')
auto_encoder.compile(optimizer='adam', loss=perceptual_loss(0.01))
auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1,
  verbose=1,)
auto_encoder.save(f'models/{model_name}_0-01.h5')

print('training with 0.1 perceptual loss')
auto_encoder.compile(optimizer='adam', loss=perceptual_loss(0.1))
auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1,
  verbose=1,)
auto_encoder.save(f'models/{model_name}_0-1.h5')

print('training with 1 perceptual loss')
auto_encoder.compile(optimizer='adam', loss=perceptual_loss(1))
auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1,
  verbose=1,)
auto_encoder.save(f'models/{model_name}_1.h5')





mobilenetv2_layers = [
  'input_2',
  'Conv1',
  'bn_Conv1',
  'Conv1_relu',
  'expanded_conv_depthwise',
  'expanded_conv_depthwise_BN',
  'expanded_conv_depthwise_relu',
  'expanded_conv_project',
  'expanded_conv_project_BN',
  'block_1_expand',
  'block_1_expand_BN',
  'block_1_expand_relu',
  'block_1_pad',
  'block_1_depthwise',
  'block_1_depthwise_BN',
  'block_1_depthwise_relu',
  'block_1_project',
  'block_1_project_BN',
  'block_2_expand',
  'block_2_expand_BN',
  'block_2_expand_relu',
  'block_2_depthwise',
  'block_2_depthwise_BN',
  'block_2_depthwise_relu',
  'block_2_project',
  'block_2_project_BN',
  'block_2_add',
  'block_3_expand',
  'block_3_expand_BN',
  'block_3_expand_relu',
  'block_3_pad',
  'block_3_depthwise',
  'block_3_depthwise_BN',
  'block_3_depthwise_relu',
  'block_3_project',
  'block_3_project_BN',
  'block_4_expand',
  'block_4_expand_BN',
  'block_4_expand_relu',
  'block_4_depthwise',
  'block_4_depthwise_BN',
  'block_4_depthwise_relu',
  'block_4_project',
  'block_4_project_BN',
  'block_4_add',
  'block_5_expand',
  'block_5_expand_BN',
  'block_5_expand_relu',
  'block_5_depthwise',
  'block_5_depthwise_BN',
  'block_5_depthwise_relu',
  'block_5_project',
  'block_5_project_BN',
  'block_5_add',
  'block_6_expand',
  'block_6_expand_BN',
  'block_6_expand_relu',
  'block_6_pad',
  'block_6_depthwise',
  'block_6_depthwise_BN',
  'block_6_depthwise_relu',
  'block_6_project',
  'block_6_project_BN',
  'block_7_expand',
  'block_7_expand_BN',
  'block_7_expand_relu',
  'block_7_depthwise',
  'block_7_depthwise_BN',
  'block_7_depthwise_relu',
  'block_7_project',
  'block_7_project_BN',
  'block_7_add',
  'block_8_expand',
  'block_8_expand_BN',
  'block_8_expand_relu',
  'block_8_depthwise',
  'block_8_depthwise_BN',
  'block_8_depthwise_relu',
  'block_8_project',
  'block_8_project_BN',
  'block_8_add',
  'block_9_expand',
  'block_9_expand_BN',
  'block_9_expand_relu',
  'block_9_depthwise',
  'block_9_depthwise_BN',
  'block_9_depthwise_relu',
  'block_9_project',
  'block_9_project_BN',
  'block_9_add',
  'block_10_expand',
  'block_10_expand_BN',
  'block_10_expand_relu',
  'block_10_depthwise',
  'block_10_depthwise_BN',
  'block_10_depthwise_relu',
  'block_10_project',
  'block_10_project_BN',
  'block_11_expand',
  'block_11_expand_BN',
  'block_11_expand_relu',
  'block_11_depthwise',
  'block_11_depthwise_BN',
  'block_11_depthwise_relu',
  'block_11_project',
  'block_11_project_BN',
  'block_11_add',
  'block_12_expand',
  'block_12_expand_BN',
  'block_12_expand_relu',
  'block_12_depthwise',
  'block_12_depthwise_BN',
  'block_12_depthwise_relu',
  'block_12_project',
  'block_12_project_BN',
  'block_12_add',
  'block_13_expand',
  'block_13_expand_BN',
  'block_13_expand_relu',
  'block_13_pad',
  'block_13_depthwise',
  'block_13_depthwise_BN',
  'block_13_depthwise_relu',
  'block_13_project',
  'block_13_project_BN',
  'block_14_expand',
  'block_14_expand_BN',
  'block_14_expand_relu',
  'block_14_depthwise',
  'block_14_depthwise_BN',
  'block_14_depthwise_relu',
  'block_14_project',
  'block_14_project_BN',
  'block_14_add',
  'block_15_expand',
  'block_15_expand_BN',
  'block_15_expand_relu',
  'block_15_depthwise',
  'block_15_depthwise_BN',
  'block_15_depthwise_relu',
  'block_15_project',
  'block_15_project_BN',
  'block_15_add',
  'block_16_expand',
  'block_16_expand_BN',
  'block_16_expand_relu',
  'block_16_depthwise',
  'block_16_depthwise_BN',
  'block_16_depthwise_relu',
  'block_16_project',
  'block_16_project_BN',
  'Conv_1',
  'Conv_1_bn',
  'out_relu']
