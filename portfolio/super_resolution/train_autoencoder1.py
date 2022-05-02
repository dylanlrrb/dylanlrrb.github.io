import tensorflow as tf
from tensorflow.keras import layers
import keras.backend as K
import sys
import pathlib
import os
from functools import reduce

# 1 = INFO messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
image_size = 448
patch_size = 224
resolution_down_factor = 0.25

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

rand_crop = layers.RandomCrop(patch_size, patch_size)

scale =  layers.Rescaling(1./255)

def train_augmentation(x):
  y = rand_crop(data_augmentation(x))
  X = reduce_resolution(y)
  return (X, scale(y))

def val_transform(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, scale(y))

def test_transform(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, scale(y))

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

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


# Build Model
tf.keras.backend.clear_session()

input_img = layers.Input(shape=(patch_size, patch_size, 3))
scaled_input_img = layers.Rescaling(1./255)(input_img)

a = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(scaled_input_img)
b = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(a)
bb = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(b)
bbb = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(bb)
c = layers.MaxPool2D(padding='same')(bbb)

l1 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(c)
l2 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
l22 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l2)
l222 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l22)
l3 = layers.MaxPool2D(padding='same')(l222)

l4 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
l5 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
l55 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l5)
l555 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l55)
l6 = layers.MaxPool2D(padding='same')(l555)

l7 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)
l8 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l7)
l88 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
l888 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l88)
l9 = layers.MaxPool2D(padding='same')(l888)

l10 = layers.SeparableConv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)
l1010 = layers.SeparableConv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l10)
l11 = layers.SeparableConv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1010)
l1111 = layers.SeparableConv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l11)

l12 = layers.UpSampling2D()(l1111)
l13 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
l14 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)
l1414 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l14)
l141414 = layers.SeparableConv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1414)

l15 = layers.add([l141414, l8])

l16 = layers.UpSampling2D()(l15)
l17 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l16)
l18 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l17)
l1818 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l18)
l181818 = layers.SeparableConv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1818)

l19 = layers.add([l181818, l5])

l20 = layers.UpSampling2D()(l19)
l21 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l20)
l22 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l21)
l2222 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l22)
l222222 = layers.SeparableConv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l2222)

l23 = layers.add([l222222, l2])

d = layers.UpSampling2D()(l23)
e = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(d)
f = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(e)
ff = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(f)
fff = layers.SeparableConv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(ff)

g = layers.add([fff, b])

decoded_image = layers.SeparableConv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(g)

auto_encoder = tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)


# Build Perceptual Loss function
def perceptual_loss(prcpt_weight, gram_weight=0.01, model="vgg", layer_maps=None):
  if model == "mobilenet":
    feature_extractor = tf.keras.applications.MobileNetV2(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet')
  if model == "vgg":
    feature_extractor = tf.keras.applications.VGG16(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet')
  feature_extractor.trainable = False
  for layer in feature_extractor.layers:
    layer.trainable=False
  
  if layer_maps is None and model == "mobilenet":
    layer_maps = ['block_2_project_BN', 'block_4_project_BN', 'block_8_project_BN', 'block_12_project_BN', 'block_16_project_BN']

  if layer_maps is None and model == "vgg":
    layer_maps = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

  loss_models = [tf.keras.Model(feature_extractor.inputs, feature_extractor.get_layer(m).output) for m in layer_maps]

  def loss_fn(y_true, y_pred):
  
    mse_loss = K.square(y_pred - y_true)
    mse_loss = K.mean(mse_loss, axis=[1,2,3])

    y_true = layers.Resizing(224, 224)(y_true)
    y_pred = layers.Resizing(224, 224)(y_pred)

    prcpt_loss = mse_loss

    for loss_model in loss_models:
      y_true_features = loss_model(y_true)
      y_pred_features = loss_model(y_pred)
      feat_loss = K.square(y_pred_features - y_true_features)
      feat_loss = K.mean(feat_loss, axis=[1,2,3])
      prcpt_loss = prcpt_loss + feat_loss

      y_true_gram = gram_matrix(y_true_features)
      y_pred_gram = gram_matrix(y_pred_features)
      gram_loss = K.square(y_pred_gram - y_true_gram)
      gram_loss = K.mean(gram_loss, axis=[1,2])
      prcpt_loss = prcpt_loss + (gram_loss * gram_weight)

    return (K.mean(prcpt_loss) * prcpt_weight) + mse_loss

  return loss_fn


model_name = 'in224_randcrop_x4zoom_plossX0-1_gramX0-01'

# Training
epochs = 6
for epoch in range(1, epochs+1):
  print(f'Epoch {epoch}/{epochs}')

  print('training with 0.1 perceptual loss')
  auto_encoder.compile(optimizer='adam', loss=perceptual_loss(0.1))
  auto_encoder.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1,)
  auto_encoder.save(f'models/{model_name}_epoch_{epoch}.h5')
