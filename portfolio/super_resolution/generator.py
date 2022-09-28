

import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Rescaling
from keras.layers import SeparableConv2D
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D


def encoder_block(layer_in, n_filters, conv_per_block=1):
  g = layer_in
  g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer='random_normal', activation=LeakyReLU(alpha=0.2))(g)
  for _ in range(conv_per_block):
    g = BatchNormalization()(g)
    g = Conv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal', activation=LeakyReLU(alpha=0.2))(g)
  return g

def decoder_block(layer_in, skip_in, n_filters, conv_per_block=1):
  g = layer_in
  g = UpSampling2D()(g)
  g = Concatenate()([g, skip_in])
  for _ in range(conv_per_block):
    g = BatchNormalization()(g)
    g = Conv2D(n_filters, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal', activation='relu')(g)
  return g


# define the standalone generator model
def define_generator(image_shape=(224,224,3), conv_per_block=3):
  in_image = Input(shape=image_shape)
  scaled_in_image = Rescaling(1./255)(in_image) # 224 x 224 x 3
  e1 = encoder_block(scaled_in_image, 32, conv_per_block=conv_per_block) # 112 x 112 x 32
  e2 = encoder_block(e1, 64, conv_per_block=conv_per_block) # 56 x 56 x 64
  e3 = encoder_block(e2, 128, conv_per_block=conv_per_block) # 28 x 28 x 128
  e4 = encoder_block(e3, 256, conv_per_block=conv_per_block) # 14 x 14 x 256
  b = BatchNormalization()(e4)
  b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer='random_normal', activation='relu')(b) # 7 x 7 x 512
  d1 = decoder_block(b, e4, 256, conv_per_block=conv_per_block)
  d2 = decoder_block(d1, e3, 256, conv_per_block=conv_per_block)
  d3 = decoder_block(d2, e2, 128, conv_per_block=conv_per_block)
  d4 = decoder_block(d3, e1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d4)
  g = Conv2D(32, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  g = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  out_image = Activation('tanh')(g)
  return Model(in_image, out_image)


def define_vgg16_generator(input_shape=(224,224,3), conv_per_block=3):
  base_model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
  layer_names = [
      'block1_pool',
      'block2_pool',
      'block3_pool',
      'block4_pool',
      'block5_pool',
    ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False

  in_image = Input(shape=input_shape)
  # scaled_in_image = Rescaling(1./255)(in_image)
  scaled_in_image = tf.keras.applications.vgg16.preprocess_input(in_image)
  e1, e2, e3, e4, e5 = down_stack(scaled_in_image)
  b = Conv2D(512, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal', activation='relu')(e5)
  d1 = decoder_block(b, e4, 512, conv_per_block=conv_per_block)
  d2 = decoder_block(d1, e3, 256, conv_per_block=conv_per_block)
  d3 = decoder_block(d2, e2, 128, conv_per_block=conv_per_block)
  d4 = decoder_block(d3, e1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d4)
  g = Conv2D(32, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  g = Conv2D(3, (4,4), strides=(1,1), padding='same', kernel_initializer='random_normal')(g)
  out_image = Activation('tanh')(g)
  return Model(in_image, out_image)


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

def define_mobilenet_generator(input_shape=(224,224,3), conv_per_block=5):
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
  layer_names = [
      'block_1_expand_relu',
      'block_3_expand_relu',
      'block_6_expand_relu',
      'block_13_expand_relu',
      'block_16_project',
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False
  
  in_image = Input(shape=input_shape)
  # scaled_in_image = Rescaling(1./255)(in_image)
  scaled_in_image = tf.keras.applications.mobilenet_v2.preprocess_input(in_image)
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
  return Model(in_image, out_image)

def upsample_attn_sep_block(layer_in, skip_in, n_filters, dropout=False, conv_per_block=1):
  g = SeparableConv2D(layer_in.shape[-1], (1,1), strides=(1,1), padding='same', activation='relu')(layer_in)
  x = SeparableConv2D(layer_in.shape[-1], (1,1), strides=(2,2), padding='same', activation='relu')(skip_in)

  a = Add()([g, x])

  a = Activation('relu')(a)

  a = SeparableConv2D(1, (1,1), strides=(1,1), padding='same', activation='sigmoid')(a)

  a = UpSampling2D()(a)

  a = Multiply()([a, skip_in])

  l_in = UpSampling2D()(layer_in)
  b = Concatenate()([l_in, a])

  b = BatchNormalization()(b)
  b = SeparableConv2D(n_filters, (3,3), strides=(1,1), padding='same', activation='relu')(b)
  if dropout:
    b = Dropout(0.5)(b)
  return b


def define_attention_generator(input_shape=(224,224,3), conv_per_block=5):
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
  layer_names = [
      'block_1_expand_relu',
      'block_3_expand_relu',
      'block_6_expand_relu',
      'block_13_expand_relu',
      'block_16_project',
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False
  
  in_image = Input(shape=input_shape)
  scaled_in_image = tf.keras.applications.mobilenet_v2.preprocess_input(in_image)
  s1, s2, s3, s4, s5 = down_stack(scaled_in_image)
  d1 = SeparableConv2D(512, (4,4), strides=(1,1), padding='same', activation='relu')(s5)
  d2 = upsample_attn_sep_block(d1, s4, 512, conv_per_block=conv_per_block)
  d3 = upsample_attn_sep_block(d2, s3, 256, conv_per_block=conv_per_block)
  d4 = upsample_attn_sep_block(d3, s2, 128, conv_per_block=conv_per_block)
  d5 = upsample_attn_sep_block(d4, s1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = SeparableConv2D(32, (4,4), strides=(1,1), padding='same')(g)
  g = SeparableConv2D(3, (4,4), strides=(1,1), padding='same')(g)
  out_image = Activation('tanh')(g)
  return Model(in_image, out_image)
 