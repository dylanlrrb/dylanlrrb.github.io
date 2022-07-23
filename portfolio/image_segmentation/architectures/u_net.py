import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Rescaling
from keras.layers import SeparableConv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Attention, MultiHeadAttention, Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D


def upsample_sep_block(layer_in, skip_in, n_filters, dropout=True, conv_per_block=1):
  g = layer_in
  g = UpSampling2D()(g)
  g = Concatenate()([g, skip_in])
  for _ in range(conv_per_block):
    g = BatchNormalization()(g)
    g = SeparableConv2D(n_filters, (3,3), strides=(1,1), padding='same', activation='relu')(g)
  if dropout:
    g = Dropout(0.5)(g)
  return g

def define_mobile_unet(input_shape=(128,128), conv_per_block=2, num_classes=81, final_activation='softmax'):
  base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(*input_shape,3), include_top=False, weights='imagenet')
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
  
  in_image = Input(shape=(*input_shape,3))
  # scaled_in_image = Rescaling(1./255)(in_image)
  scaled_in_image = tf.keras.applications.mobilenet_v2.preprocess_input(in_image)
  s1, s2, s3, s4, s5 = down_stack(scaled_in_image)
  d1 = SeparableConv2D(512, (3,3), strides=(1,1), padding='same', activation='relu')(s5)
  d2 = upsample_sep_block(d1, s4, 512, conv_per_block=conv_per_block)
  d3 = upsample_sep_block(d2, s3, 256, conv_per_block=conv_per_block)
  d4 = upsample_sep_block(d3, s2, 128, conv_per_block=conv_per_block)
  d5 = upsample_sep_block(d4, s1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = BatchNormalization()(g)
  g = SeparableConv2D(num_classes, (3,3), strides=(1,1), padding='same')(g)
  out_image = Activation(final_activation)(g)
  return Model(in_image, out_image)



def upsample_block(layer_in, skip_in, n_filters, dropout=True, conv_per_block=1):
  g = layer_in
  g = UpSampling2D()(g)
  g = Concatenate()([g, skip_in])
  for _ in range(conv_per_block):
    g = BatchNormalization()(g)
    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same', activation='relu')(g)
  if dropout:
    g = Dropout(0.5)(g)
  return g

def define_vgg_unet(input_shape=(128,128), conv_per_block=1, num_classes=81, final_activation='softmax'):
  base_model = tf.keras.applications.VGG16(input_shape=(*input_shape,3), include_top=False, weights='imagenet')
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

  in_image = Input(shape=(*input_shape,3))
  scaled_in_image = Rescaling(1./255)(in_image)
  e1, e2, e3, e4, e5 = down_stack(scaled_in_image)
  d1 = Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu')(e5)
  d2 = upsample_block(d1,  e4, 512, conv_per_block=conv_per_block)
  d3 = upsample_block(d2, e3, 256, conv_per_block=conv_per_block)
  d4 = upsample_block(d3, e2, 128, conv_per_block=conv_per_block)
  d5 = upsample_block(d4, e1, 64,  conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = BatchNormalization()(g)
  g = Conv2D(num_classes, (3,3), strides=(1,1), padding='same')(g)
  out_image = Activation(final_activation)(g)
  return Model(in_image, out_image)
