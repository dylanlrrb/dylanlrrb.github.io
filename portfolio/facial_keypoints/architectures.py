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

# def sep_conv_block(layer_in, n_filters, kernel_size, activation, batch_norm, conv_per_block, depthwise_initializer, pointwise_initializer, dropout=None):
#   x = layer_in
#   for _ in range(conv_per_block):
#     if batch_norm:
#       x = BatchNormalization()(x)
#     x = SeparableConv2D(n_filters, kernel_size, padding='same', activation=activation, depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer)(x)
#   x = MaxPool2D()(x)
#   x = Dropout(dropout)(x)
#   return x


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
  in_image = Input(shape=(*input_shape,1))
  x = in_image
  n_filters = 32
  for _ in range(blocks):
    x = sep_conv_block(layer_in=x, n_filters=n_filters, kernel_size=kernel_size, activation=activation, batch_norm=batch_norm, conv_per_block=conv_per_block, depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer, dropout=dropout)
    n_filters = min(n_filters * 2, 512)
  # x = SeparableConv2D(1, kernel_size, padding='same', depthwise_initializer=depthwise_initializer, pointwise_initializer=pointwise_initializer)(x)
  x = Conv2D(1, kernel_size, padding='same')(x)
  x = Flatten()(x)
  x = BatchNormalization()(x)
  for _ in range(dense_layers):
    x = Dense(100)(x)
    x = BatchNormalization()(x)
  out = Dense(output_size, activation='tanh')(x)
  return Model(in_image, out)
