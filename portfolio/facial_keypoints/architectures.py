import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Rescaling
from keras.layers import SeparableConv2D
from keras.layers import UpSampling2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Attention, MultiHeadAttention, Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D


def sep_conv_block(layer_in, n_filters, kernel_size, activation, batch_norm, conv_per_block, depthwise_initializer, pointwise_initializer, dropout=None):
  x = layer_in
  x = SeparableConv2D(n_filters, kernel_size, padding='same', depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer)(x)
  if batch_norm:
      x = BatchNormalization()(x)
  for _ in range(1, conv_per_block):
    x = Activation(activation)(x)
    x = SeparableConv2D(n_filters, kernel_size, padding='same', depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer)(x)
    if batch_norm:
      x = BatchNormalization()(x)
  x = Concatenate()([x, layer_in])
  x = Activation(activation)(x)
  x = MaxPool2D()(x)
  x = Dropout(dropout)(x)
  return x


def sep_conv_net(input_shape, output_size, blocks, dense_layers, conv_per_block, kernel_size, activation, batch_norm, dropout, depthwise_initializer, pointwise_initializer):
  in_image = Input(shape=(*input_shape,3))
  x = Rescaling(225.)(in_image)
  n_filters = 32
  for _ in range(blocks):
    x = sep_conv_block(layer_in=x, n_filters=n_filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, conv_per_block=conv_per_block, depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer, dropout=dropout)
    n_filters = min(n_filters * 2, 512)
  for _ in range(5):
    x = SeparableConv2D(output_size, kernel_size=kernel_size, strides=(2,2), padding='same', activation=activation)(x)
    x = BatchNormalization()(x)
  x = Flatten()(x)
  out = Dense(output_size, activation='tanh')(x)
  return Model(in_image, out)


def mobile_net_backbone(input_shape, output_size, dropout, trainable=False):
  feature_extractor = tf.keras.applications.mobilenet_v2.MobileNetV2(
                          input_shape=(*input_shape, 3),
                          include_top=False,
                          weights='imagenet')
  feature_extractor.trainable = trainable

  in_image = Input(shape=(*input_shape,3))
  x = tf.keras.applications.mobilenet_v2.preprocess_input(in_image)
  x = feature_extractor(x)
  for _ in range(3):
    x = Conv2D(output_size, kernel_size=4, strides=(2,2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
  x = Flatten()(x)
  out = Dense(output_size, activation='tanh')(x)
  return Model(in_image, out)

def resnet_50_backbone(input_shape, output_size, dense_layers, dropout, trainable=False):
  feature_extractor = tf.keras.applications.resnet50.ResNet50(input_shape=(*input_shape, 3),
                                                        include_top=False,
                                                        weights='imagenet')
  feature_extractor.trainable = trainable

  in_image = Input(shape=(*input_shape,3))
  x = tf.keras.applications.resnet50.preprocess_input(in_image)
  x = feature_extractor(x)
  x = Conv2D(1, 4, padding='same')(x)
  x = Flatten()(x)
  for _ in range(dense_layers):
    x = Dense(200)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
  out = Dense(output_size, activation='tanh')(x)
  return Model(in_image, out)

