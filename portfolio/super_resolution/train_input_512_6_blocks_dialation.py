import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
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
image_size = 512
resolotion_down_factor = 0.2

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
  layers.RandomRotation(0.2),
])

reduce_resolution = tf.keras.Sequential([
  layers.Resizing(int(image_size*resolotion_down_factor), int(image_size*resolotion_down_factor)),
  layers.Resizing(image_size, image_size),
])

scale = layers.Rescaling(1./255)

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

epochs = 5 if hyperparameter_tuning else 1

# Build Model
model_name = sys.argv[0].replace('train_', '').replace('.py', '') + ('' if hyperparameter_tuning else '_full_train')

tf.keras.backend.clear_session()

input_img = layers.Input(shape=(image_size, image_size, 3))

l1 = layers.Conv2D(64, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
l2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
l3 = layers.MaxPool2D(padding='same')(l2)


l4 = layers.Conv2D(128, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
l5 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
l6 = layers.MaxPool2D(padding='same')(l5)

l7 = layers.Conv2D(256, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)
l8 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l7)
l9 = layers.MaxPool2D(padding='same')(l8)

l10 = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)

l11 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l10)
l12 = layers.Conv2D(256, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l11)
l13 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)

l14 = layers.add([l13, l8])

l15 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l14)
l16 = layers.Conv2D(128, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)
l17 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l16)

l18 = layers.add([l17, l5])

l19 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l18)
l20 = layers.Conv2D(64, (3, 3), dilation_rate=(2,2), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l19)
l21 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l20)

l22 = layers.add([l21, l2])

decoded_image = layers.Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l22)

auto_encoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)

auto_encoder.compile(
  optimizer='adam',
  loss='mean_squared_error')

# Training
tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=f'logs/{model_name}',
  histogram_freq=1)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=f'models/{model_name}.h5',
  monitor='val_loss',
  mode='min',
  save_best_only=True)

history = auto_encoder.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[model_checkpoint_callback, tensorboard_callback])

# Test Model
tf.keras.backend.clear_session()
best_model = tf.keras.models.load_model(f'models/{model_name}.h5')
print('Evaluation:')
best_model.evaluate(test_ds)
