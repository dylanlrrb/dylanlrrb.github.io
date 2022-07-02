import os
import io
import gc
from math import ceil, inf
import pickle as pkl
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import RandomFlip, RandomRotation, Resizing, RandomCrop, Rescaling, BatchNormalization, Input
from keras.models import Model
from dataset_helpers import split_training_image_dataset
from generator import define_generator, define_mobilenet_generator, define_vgg16_generator
from discriminator import define_discriminator
from perceptual_loss import PerceptualLoss
import wandb

# colors of predicted image are off sometimes, maybe save model based on loses mse loss?

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 4
image_size = 500
patch_size = 224
resolution_down_factor = 0.25


data_augmentation = tf.keras.Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),])

reduce_resolution = tf.keras.Sequential([
  Resizing(int(patch_size*resolution_down_factor), int(patch_size*resolution_down_factor)),
  Resizing(patch_size, patch_size),])

rand_crop = RandomCrop(patch_size, patch_size)

tanh_scale =  Rescaling(1./127.5, offset=-1)
undo_tanh_scale =  lambda x: ((x + 1) * 127.5).astype(int)

def train_transforms(x):
  y = rand_crop(data_augmentation(x))
  X = reduce_resolution(y)
  return (X, tanh_scale(y))

def val_transforms(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, tanh_scale(y))

def test_transforms(x):
  y = rand_crop(x)
  X = reduce_resolution(y)
  return (X, tanh_scale(y))


train_ds, val_ds, test_ds = split_training_image_dataset(
  'flickr30k_images',
  'https://datasets-349058029.s3.us-west-2.amazonaws.com/flickr/flickr30k_images.zip',
  train_transforms,
  val_transforms,
  test_transforms,
  labels=None,
  data_subsample=1,
  batch_size=batch_size,
  image_size=(image_size,image_size),
)

# Logging helpers
def create_comparison_plot(model, img_path, epoch):
    img = np.asarray(Image.open(f'assets/{img_path}'))
    target = np.array([cv2.resize(img, (patch_size, patch_size))])
    input = reduce_resolution(target)
    out = undo_tanh_scale(model.predict(input))
    
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.text(0., 1., f'{epoch}'.zfill(2),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', edgecolor='none'))
    ax1.imshow(np.squeeze(np.array(input)).astype(int))
    ax1.title.set_text('Input')
    ax1.axis('off')
    ax2.imshow(np.squeeze(np.array(out)).astype(int))
    ax2.title.set_text('Prediction')
    ax2.axis('off')
    ax3.imshow(target.squeeze().astype(int))
    ax3.title.set_text('Ground Truth')
    ax3.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return figure

def plot_to_img(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  return image

def pickle_img_seq(model_name, img_seq, name):
  with open(f'train_samples/{model_name}_{name}.pkl', 'wb') as f:
    pkl.dump(img_seq, f)

samples = [{'name': 'baboon', 'extension': 'jpg', 'description': 'Baboon', 'train_samples': []},
          {'name': 'butterfly', 'extension': 'jpg', 'description': 'Butterfly', 'train_samples': []},
          {'name': 'castle', 'extension': 'jpg', 'description': 'Castle', 'train_samples': []},
          {'name': 'eye', 'extension': 'jpg', 'description': 'Eye', 'train_samples': []},
          {'name': 'lion', 'extension': 'jpg', 'description': 'Lion', 'train_samples': []},
          {'name': 'woman', 'extension': 'jpg', 'description': 'Woman', 'train_samples': []},]

class GAN():
  def __init__(self, image_shape, d_lr=2e-4, g_lr=2e-4):
    # Build Models

    self.generator = define_mobilenet_generator(image_shape)
    # self.generator = define_vgg16_generator(image_shape)
    # self.generator = define_generator(image_shape)

    self.discriminator = define_discriminator(image_shape)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=0.5)
    self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, loss_weights=[0.5])
    
    # Build composite model
    for layer in self.discriminator.layers:
      if not isinstance(layer, BatchNormalization):
        layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = self.generator(in_src)
    dis_out = self.discriminator([in_src, gen_out])

    self.composite_gan = Model(in_src, [dis_out, gen_out])
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=0.5)
    self.composite_gan.compile(optimizer=generator_optimizer,
                              loss=['binary_crossentropy', PerceptualLoss()],
                              metrics=['mse'])



tf.keras.backend.clear_session()

sr_gan = GAN(image_shape=(patch_size, patch_size, 3))

generate_real_samples = iter(train_ds)
generate_fake_samples = lambda samples: sr_gan.generator.predict(samples)
def generate_discriminator_patches(samples_in_batch, patch_shape):
  discriminator_real = tf.ones((samples_in_batch, patch_shape, patch_shape, 1))
  discriminator_fake = tf.zeros((samples_in_batch, patch_shape, patch_shape, 1))
  return discriminator_real, discriminator_fake

# Training
model_name = 'mobilenet_4x'
epochs = 30
batches_per_epoch = 500
iterations = epochs * batches_per_epoch
log_per_n_iterations = 100

wandb.init(project="super-resolution", name=model_name, config={
  'conv_per_block': 4,
  "d_lr": 2e-4,
  "d_beta": 0.5,
  "g_lr": 2e-4,
  "g_beta": 0.5,
  "epochs": epochs,
  "batches_per_epoch": batches_per_epoch,
  "log_per_n_iterations": log_per_n_iterations,
  "batch_size": batch_size,
  "patch_size": patch_size,
  "in_image_size": image_size,
  "resolution_down_factor": resolution_down_factor,
})

print(f'Epoch 1 / {epochs}')
progbar = tf.keras.utils.Progbar(batches_per_epoch)
best_mse_loss = inf
best_p_loss = inf

for i in range(iterations):

  accumulated_d_loss = []
  accumulated_g_loss = []
  accumulated_p_loss = []
  accumulated_mse_loss = []

  X, real_y = next(generate_real_samples)
  generated_y = generate_fake_samples(X)
  samples_in_batch = len(X)
  discriminator_patch_shape = sr_gan.discriminator.output_shape[1]
  discriminator_real, discriminator_fake = generate_discriminator_patches(samples_in_batch, discriminator_patch_shape)

  d_loss1 = sr_gan.discriminator.train_on_batch([X, real_y], discriminator_real)
  d_loss2 = sr_gan.discriminator.train_on_batch([X, generated_y], discriminator_fake)
  d_loss_total = d_loss1 + d_loss2
  g_loss_total, g_loss, p_loss, _, mse_loss = sr_gan.composite_gan.train_on_batch(X, [discriminator_real, real_y])

  accumulated_d_loss.append(d_loss_total)
  accumulated_g_loss.append(g_loss)
  accumulated_p_loss.append(p_loss)
  accumulated_mse_loss.append(mse_loss)

  progbar.update(i % batches_per_epoch, values=[('d_loss', d_loss_total), ('g_loss', g_loss), ('p_loss', p_loss), ('mse', mse_loss)])

  if (i+1) % log_per_n_iterations == 0:
    accumulated_d_loss_mean = np.mean(accumulated_d_loss)
    accumulated_g_loss_mean = np.mean(accumulated_g_loss)
    accumulated_p_loss_mean = np.mean(accumulated_p_loss)
    accumulated_mse_loss_mean = np.mean(accumulated_mse_loss)
    wandb.log({
      "discriminator_loss" : accumulated_d_loss_mean,
      "generator_loss" : accumulated_g_loss_mean,
      "perceptual_loss": accumulated_p_loss_mean,
      "mse_loss": accumulated_mse_loss_mean})
    if accumulated_mse_loss_mean < best_mse_loss:
      print(f'\n\nnew mse_loss of {round(accumulated_mse_loss_mean, 3)} better than prev best {round(best_mse_loss, 3)}, Saving...\n\n')
      sr_gan.generator.save(f'models/{model_name}_BEST-MSE.h5')
      best_mse_loss = accumulated_mse_loss_mean
    if accumulated_p_loss_mean < best_p_loss:
      print(f'\n\nnew p_loss of {round(accumulated_p_loss_mean, 3)} better than prev best {round(best_p_loss, 3)}, Saving...\n\n')
      sr_gan.generator.save(f'models/{model_name}_BEST-P.h5')
      best_p_loss = accumulated_p_loss_mean
    accumulated_d_loss = []
    accumulated_g_loss = []
    accumulated_p_loss = []
    accumulated_mse_loss = []
    gc.collect()
    

  if (i+1) % batches_per_epoch == 0:
    progbar.update(i % batches_per_epoch, values=[('d_loss', d_loss_total), ('g_loss', g_loss), ('p_loss', p_loss), ('mse', mse_loss)], finalize=True)
    epoch_num = ceil((i+1) / batches_per_epoch) + 1
    print('' if epoch_num > epochs else f'\nEpoch {epoch_num} / {epochs}')
    progbar = tf.keras.utils.Progbar(batches_per_epoch)
  
    for sample in samples:
      plot = create_comparison_plot(sr_gan.generator, f"{sample['name']}.{sample['extension']}", epoch_num-1)
      img = plot_to_img(plot)
      sample['train_samples'].append(img)
      pickle_img_seq(model_name, sample['train_samples'], sample['name'])
      wandb.log({sample['description']: wandb.Image(img)})

    gc.collect()

sr_gan.generator.save(f'models/{model_name}_FINAL.h5')
    