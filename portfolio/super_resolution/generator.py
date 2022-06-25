

import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Rescaling
from keras.layers import SeparableConv2D
from keras.layers import UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D


def define_encoder_block(layer_in, n_filters, batchnorm=True, maxpool=True, conv_per_block=1):
  for i in range(conv_per_block):
    g = SeparableConv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal', activation=LeakyReLU(alpha=0.2), use_bias=(not batchnorm))(layer_in)
    if batchnorm:
      g = BatchNormalization()(g)
  if maxpool:
    g = MaxPool2D(padding='same')(g)
  g = LeakyReLU(alpha=0.2)(g)
  return g
 

def decoder_block(layer_in, skip_in, n_filters, upsample=True, dropout=True, conv_per_block=1):
  
  g = layer_in
  if upsample:
    g = UpSampling2D()(g)
  for i in range(conv_per_block):
    g = SeparableConv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal', activation='relu', use_bias=False)(g)
    g = BatchNormalization()(g)
    if dropout:
      g = Dropout(0.5)(g)
  g = Concatenate()([g, skip_in])
  g = Activation('relu')(g)
  return g


# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
  in_image = Input(shape=image_shape)
  scaled_in_image = Rescaling(1./255)(in_image)
  e1 = define_encoder_block(scaled_in_image, 64, conv_per_block=3, batchnorm=False)
  e2 = define_encoder_block(e1, 128, conv_per_block=3)
  e3 = define_encoder_block(e2, 256, conv_per_block=3)
  e4 = define_encoder_block(e3, 512, conv_per_block=3)
  e5 = define_encoder_block(e4, 512, conv_per_block=3)
  e6 = define_encoder_block(e5, 512, conv_per_block=3)
  e7 = define_encoder_block(e6, 512, conv_per_block=3)
  b = SeparableConv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer='random_normal', activation='relu')(e7)
  d1 = decoder_block(b, e7, 512, conv_per_block=3)
  d2 = decoder_block(d1, e6, 512, conv_per_block=3)
  d3 = decoder_block(d2, e5, 512, conv_per_block=3)
  d4 = decoder_block(d3, e4, 512, conv_per_block=3, dropout=False)
  d5 = decoder_block(d4, e3, 256, conv_per_block=3, dropout=False)
  d6 = decoder_block(d5, e2, 128, conv_per_block=3, dropout=False)
  d7 = decoder_block(d6, e1, 64, conv_per_block=3, dropout=False)
  g = UpSampling2D()(d7)
  g = SeparableConv2D(32, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  g = SeparableConv2D(32, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  g = SeparableConv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  out_image = Activation('tanh')(g)
  model = Model(in_image, out_image)
  return model




def upsample_sep_block(layer_in, skip_in, n_filters, dropout=False, conv_per_block=1):
  g = layer_in
  g = UpSampling2D()(g)
  g = Concatenate()([g, skip_in])
  for _ in range(conv_per_block):
    g = BatchNormalization()(g)
    g = SeparableConv2D(n_filters, (4,4), strides=(1,1), padding='same', activation='relu')(g)
  if dropout:
    g = Dropout(0.5)(g)
  return g

def define_mobilenet_generator(input_shape=(128,128,3), conv_per_block=5):
  base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False
  
  in_image = Input(shape=input_shape)
  scaled_in_image = Rescaling(1./255)(in_image)
  s1, s2, s3, s4, s5 = down_stack(scaled_in_image)
  d1 = SeparableConv2D(512, (4,4), strides=(1,1), padding='same', activation='relu')(s5)
  d2 = upsample_sep_block(d1, s4, 512, conv_per_block=conv_per_block)
  d3 = upsample_sep_block(d2, s3, 256, conv_per_block=conv_per_block)
  d4 = upsample_sep_block(d3, s2, 128, conv_per_block=conv_per_block)
  d5 = upsample_sep_block(d4, s1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = SeparableConv2D(32, (4,4), strides=(1,1), padding='same')(g)
  g = SeparableConv2D(3, (4,4), strides=(1,1), padding='same')(g)
  out_image = Activation('tanh')(g)
  model = Model(in_image, out_image)
  return model
 