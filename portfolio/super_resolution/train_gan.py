import os
import io
import gc
from math import ceil
import pickle as pkl
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import RandomFlip, RandomRotation, Resizing, RandomCrop, Rescaling, BatchNormalization, Input
from keras.models import Model
from dataset_helpers import split_training_image_dataset
from generator import define_generator, define_mobilenet_generator
from discriminator import define_discriminator
from perceptual_loss import PerceptualLoss

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


# Build Models
tf.keras.backend.clear_session()
image_shape = (patch_size, patch_size, 3)
# generator = define_generator(image_shape)
generator = define_mobilenet_generator(image_shape)
discriminator = define_discriminator(image_shape)

# generator.summary()

# Build composite model
for layer in discriminator.layers:
  if not isinstance(layer, BatchNormalization):
    layer.trainable = False
in_src = Input(shape=image_shape)
gen_out = generator(in_src)
dis_out = discriminator([in_src, gen_out])

composite_gan = Model(in_src, [dis_out, gen_out])
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
composite_gan.compile(loss=['binary_crossentropy', PerceptualLoss()], optimizer=opt)

# Logging helpers
def create_comparison_plot(model, img_path, epoch):
    img = np.asarray(Image.open(f'assets/{img_path}'))
    target = np.array([cv2.resize(img, (patch_size, patch_size))])
    input = reduce_resolution(target)
    out = undo_tanh_scale(model.predict(input))
    
    figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.text(0., 1., f'{epoch}',
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

def log_img_to_tensorboard(epoch, img, img_description, model_name):
    file_writer = tf.summary.create_file_writer(f'logs/{model_name}/sample_images')
    with file_writer.as_default():
        tf.summary.image(f'{model_name}/{img_description}', tf.expand_dims(img, 0), step=epoch)
    file_writer.flush()

def pickle_img_seq(model_name, img_seq, name):
    with open(f'train_samples/{model_name}_{name}.pkl', 'wb') as f:
      pkl.dump(img_seq, f)

samples = [{'name': 'baboon', 'extension': 'jpg', 'description': 'Baboon', 'train_samples': []},
          {'name': 'butterfly', 'extension': 'jpg', 'description': 'Butterfly', 'train_samples': []},
          {'name': 'castle', 'extension': 'jpg', 'description': 'castle', 'train_samples': []},
          {'name': 'eye', 'extension': 'jpg', 'description': 'Eye', 'train_samples': []},
          {'name': 'lion', 'extension': 'jpg', 'description': 'Lion', 'train_samples': []},
          {'name': 'woman', 'extension': 'jpg', 'description': 'Woman', 'train_samples': []},]


# Training
# model_name = 'gan_in256_4Xzoom_plossX0-1_full_train'
model_name = 'gan_in224_4Xzoom_plossX0-1_monilenet_backbone_conv5'
epochs = 50
batches_per_epoch = 500
iterations = epochs * batches_per_epoch
log_per_n_iterations = 100


generate_real_samples = iter(train_ds)
generate_fake_samples = lambda samples: generator.predict(samples)
def generate_discriminator_patches(samples_in_batch, patch_shape):
  discriminator_real = tf.ones((samples_in_batch, patch_shape, patch_shape, 1))
  discriminator_fake = tf.zeros((samples_in_batch, patch_shape, patch_shape, 1))
  return discriminator_real, discriminator_fake

summary_writer_discriminator = tf.summary.create_file_writer(f'logs/discriminator_loss')
summary_writer_generator = tf.summary.create_file_writer(f'logs/generator_loss')
summary_writer_perceptual = tf.summary.create_file_writer(f'logs/perceptual_loss')

print(f'Epoch 1 / {epochs}')
progbar = tf.keras.utils.Progbar(batches_per_epoch)

for i in range(iterations):

  accumulated_d_loss = []
  accumulated_g_loss = []
  accumulated_p_loss = []

  X, real_y = next(generate_real_samples)
  generated_y = generate_fake_samples(X)
  samples_in_batch = len(X)
  discriminator_patch_shape = discriminator.output_shape[1]
  discriminator_real, discriminator_fake = generate_discriminator_patches(samples_in_batch, discriminator_patch_shape)

  d_loss1 = discriminator.train_on_batch([X, real_y], discriminator_real)
  d_loss2 = discriminator.train_on_batch([X, generated_y], discriminator_fake)
  d_loss_total = d_loss1 + d_loss2
  g_loss_total, g_loss, p_loss = composite_gan.train_on_batch(X, [discriminator_real, real_y])

  accumulated_d_loss.append(d_loss_total)
  accumulated_g_loss.append(g_loss)
  accumulated_p_loss.append(p_loss)

  progbar.update(i % batches_per_epoch, values=[('d_loss', d_loss_total), ('g_loss', g_loss), ('p_loss', p_loss)])

  if (i+1) % log_per_n_iterations == 0:
    with summary_writer_discriminator.as_default():
      tf.summary.scalar(f"{model_name}/gan", np.mean(accumulated_d_loss), step=i)
    summary_writer_discriminator.flush()
    with summary_writer_generator.as_default():
      tf.summary.scalar(f"{model_name}/gan", np.mean(accumulated_g_loss), step=i)
    summary_writer_generator.flush()
    with summary_writer_perceptual.as_default():
      tf.summary.scalar(f"{model_name}/perceptual_loss", np.mean(accumulated_p_loss), step=i)
    summary_writer_perceptual.flush()
    accumulated_d_loss = []
    accumulated_g_loss = []
    accumulated_p_loss = []
    gc.collect()
    

  if (i+1) % batches_per_epoch == 0:
    progbar.update(i % batches_per_epoch, values=[('d_loss', d_loss_total), ('g_loss', g_loss), ('p_loss', p_loss)], finalize=True)
    generator.save(f'models/{model_name}/iteration_{i}.h5')
    epoch_num = ceil((i+1) / batches_per_epoch) + 1
  
    for sample in samples:
      plot = create_comparison_plot(generator, f"{sample['name']}.{sample['extension']}", epoch_num-1)
      img = plot_to_img(plot)
      sample['train_samples'].append(img)
      pickle_img_seq(model_name, sample['train_samples'], sample['name'])
      log_img_to_tensorboard(epoch_num-1, img, sample['description'], model_name)

    print('' if epoch_num > epochs else f'\nEpoch {epoch_num} / {epochs}')
    progbar = tf.keras.utils.Progbar(batches_per_epoch)
    gc.collect()
    