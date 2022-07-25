import argparse
import wandb
import os
from architectures import sep_conv_net, mobile_net_backbone, resnet_50_backbone
from preprocess import dataGenerator, to_tf_dataset
from callbacks import model_checkpoint_callback, CustomCallback, reduce_lr, GarbageCollection
from train import custom_loss
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from wandb.keras import WandbCallback
import keras.backend as K
from keras.layers import BatchNormalization

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_NAME = 'fine_tune_custom_loss-5'
BASE_MODEL_NAME = 'mobilenet_backbone_full_non_trainable'

hyperparameter_defaults = dict(
    image_size=224,
    output_size = 136,
    epochs = 200,
    loss = 'custom',
    batch_size = 32,
    steps_per_epoch = 100,
    validation_steps = 10,
    optimizer = 'Adam',
    learning_rate = 1e-5,
    lr_decay_factor = 0.2,
    lr_decay_patience = 20,
    lr_decay_cooldown = 20,
  )


IMAGE_SIZE = (hyperparameter_defaults["image_size"],hyperparameter_defaults["image_size"])
OUTPUT_SIZE = hyperparameter_defaults["output_size"]

def main(project_dir, dataset_dir, sweep):
  wandb.init(config=hyperparameter_defaults, name=MODEL_NAME, project="facial_keypoints", dir=project_dir)
  config = wandb.config
  print('configs for this sweep:', config)

  train_generator = dataGenerator(f'{dataset_dir}/training_frames_keypoints.csv',
                                  f'{dataset_dir}/training',
                                  f'{project_dir}/haarcascade_frontalface_default.xml',
                                  training=True,
                                  input_size=IMAGE_SIZE,
                                  batch_size=config.batch_size,
                                  shuffle=True)
  train_dataset = to_tf_dataset(train_generator,
                                batch_size=config.batch_size,
                                img_size=IMAGE_SIZE,
                                output_size=OUTPUT_SIZE)

  test_generator = dataGenerator( f'{dataset_dir}/test_frames_keypoints.csv',
                                  f'{dataset_dir}/test', 
                                  f'{project_dir}/haarcascade_frontalface_default.xml',
                                  training=False,
                                  input_size=IMAGE_SIZE,
                                  batch_size=config.batch_size,
                                  shuffle=True)
  test_dataset = to_tf_dataset( test_generator,
                                batch_size=config.batch_size,
                                img_size=IMAGE_SIZE,
                                output_size=OUTPUT_SIZE)

  model = tf.keras.models.load_model(f'{project_dir}/models/{BASE_MODEL_NAME}.h5', compile=False)
  # unfreeze backbone model
  for layer in model.layers:
    if not isinstance(layer, BatchNormalization):
      layer.trainable = True

  model.compile(loss=(custom_loss if config.loss == 'custom' else config.loss),
                optimizer=getattr(tf.keras.optimizers, config.optimizer)(learning_rate=config.learning_rate))

  model.summary()
  

  # extra_train_generator = dataGenerator( f'{dataset_dir}/test_frames_keypoints.csv',
  #                                 f'{dataset_dir}/test', 
  #                                 f'{project_dir}/haarcascade_frontalface_default.xml',
  #                                 training=True,
  #                                 input_size=IMAGE_SIZE,
  #                                 batch_size=config.batch_size,
  #                                 shuffle=True)
  # extra_train_dataset = to_tf_dataset( extra_train_generator,
  #                               batch_size=config.batch_size,
  #                               img_size=IMAGE_SIZE,
  #                               output_size=OUTPUT_SIZE)

  model.fit(train_dataset, epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            validation_steps=config.validation_steps,
            validation_data=test_dataset,
            callbacks=[ WandbCallback(save_model=False),
                        reduce_lr(config.lr_decay_factor, config.lr_decay_patience, config.lr_decay_cooldown),
                        *([model_checkpoint_callback(project_dir, MODEL_NAME), CustomCallback(project_dir, MODEL_NAME, IMAGE_SIZE, f'{project_dir}/haarcascade_frontalface_default.xml')] if not sweep else [GarbageCollection()])
                      ])


if __name__ == '__main__':
  my_parser = argparse.ArgumentParser(description='Training script for facial keypoint CNN')

  my_parser.add_argument('--sweep',
                        '-s',
                        action='store_true',
                        help='flag if running as a sweep')

  args = my_parser.parse_args()

  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  for gpu in tf.config.list_physical_devices('GPU'):
    print(gpu)

  os.chdir('/')

  IN_COLAB = os.path.isdir("content/drive/MyDrive")
  print('In Colab:', IN_COLAB)

  PROJECT_DIR = "content/drive/MyDrive/dylanlrrb.github.io/portfolio/facial_keypoints" if IN_COLAB else 'tf/notebooks/portfolio/facial_keypoints'
  DATASET_DIR = "content/facial_keypoints" if IN_COLAB else 'tf/notebooks/portfolio/facial_keypoints/data'
  SWEEP = args.sweep

  main(PROJECT_DIR, DATASET_DIR, SWEEP)