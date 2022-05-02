import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import applications

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


def PerceptualLoss(prcpt_weight=0.1, gram_weight=0.01, model="vgg", layer_maps=None):
  if model == "mobilenet":
    feature_extractor = applications.MobileNetV2(
      input_shape=(224, 224, 3),
      include_top=False,
      weights='imagenet')
  if model == "vgg":
    feature_extractor = applications.VGG16(
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