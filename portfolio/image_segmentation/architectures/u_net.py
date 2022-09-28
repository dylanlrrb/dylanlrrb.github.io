import tensorflow as tf
from keras.models import Model
from keras.models import Input
from keras.layers import Rescaling
from keras.layers import SeparableConv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Attention, MultiHeadAttention, Reshape, Multiply, Activation, Add, Subtract
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras import backend as K
from keras.layers import Lambda

class BatAttn(tf.keras.layers.Layer): 
    def __init__(self):    
        super(BatAttn, self).__init__()
        
    def build(self, input_shape):
      dim = input_shape[-1]
      num_vectors = input_shape[-2]
      num_units = 1
      self.W = self.add_weight(shape=(dim,num_units), initializer='normal')
      self.b = self.add_weight(shape=(num_vectors,num_units), initializer='zero')
      super(BatAttn, self).build(input_shape)
        
    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        return K.sum(output, axis=1)


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

def upsample_att_sep_block(layer_in, skip_in, n_filters, dropout=True, conv_per_block=1):
  g = SeparableConv2D(layer_in.shape[-1], (1,1), strides=(1,1), padding='same', activation='relu')(layer_in)
  x = SeparableConv2D(layer_in.shape[-1], (1,1), strides=(2,2), padding='same', activation='relu')(skip_in)

  a = Add()([g, x])

  a = Activation('relu')(a)

  a = SeparableConv2D(1, (1,1), strides=(1,1), padding='same', activation='sigmoid')(a)

  a = UpSampling2D()(a)
  
  # def inverse(tensor):
  #   return K.abs(tensor - 1)
  # a = Lambda(inverse)(a)

  skip_in = Multiply()([a, skip_in])
  layer_in = UpSampling2D()(layer_in)

  b = Concatenate()([layer_in, skip_in])

  b = BatchNormalization()(b)
  b = SeparableConv2D(n_filters, (3,3), strides=(1,1), padding='same', activation='relu')(b)
  if dropout:
    b = Dropout(0.5)(b)
  return b, a

def define_attention_mobile_unet(input_shape=(128,128), conv_per_block=2, num_classes=81, final_activation='softmax'):
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
  scaled_in_image = tf.keras.applications.mobilenet_v2.preprocess_input(in_image)
  s1, s2, s3, s4, s5 = down_stack(scaled_in_image)
  d1 = SeparableConv2D(576, (3,3), strides=(1,1), padding='same', activation='relu')(s5)
  d2, a1 = upsample_att_sep_block(d1, s4, 192, conv_per_block=conv_per_block)
  d3, a2 = upsample_att_sep_block(d2, s3, 144, conv_per_block=conv_per_block)
  d4, a3 = upsample_att_sep_block(d3, s2, 96, conv_per_block=conv_per_block)
  d5, a4 = upsample_att_sep_block(d4, s1, 64, conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = BatchNormalization()(g)
  g = SeparableConv2D(num_classes, (3,3), strides=(1,1), padding='same')(g)
  out_image = Activation(final_activation)(g)
  return Model(in_image, out_image), Model(in_image, [a1, a2, a3, a4])



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

def attn_upsample_block(layer_in, skip_in, n_filters, dropout=True, conv_per_block=1):
  g = layer_in
  g = UpSampling2D()(g)
  
  skip_in = Conv2D(g.shape[-1], (3,3), strides=(1,1), padding='same', activation='relu')(skip_in)
  g_x = Conv2D(g.shape[-1], (3,3), strides=(1,1), padding='same', activation='relu')(g)

  g_x = Add()([g_x, skip_in])
  g_x = Activation('sigmoid')(g_x)

  g = Multiply()([g_x, g])
  g = Activation('relu')(g)

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
  d2 = attn_upsample_block(d1,  e4, 512, conv_per_block=conv_per_block)
  d3 = attn_upsample_block(d2, e3, 256, conv_per_block=conv_per_block)
  d4 = attn_upsample_block(d3, e2, 128, conv_per_block=conv_per_block)
  d5 = attn_upsample_block(d4, e1, 64,  conv_per_block=conv_per_block)
  g = UpSampling2D()(d5)
  g = BatchNormalization()(g)
  g = Conv2D(num_classes, (3,3), strides=(1,1), padding='same')(g)
  out_image = Activation(final_activation)(g)
  return Model(in_image, out_image)
