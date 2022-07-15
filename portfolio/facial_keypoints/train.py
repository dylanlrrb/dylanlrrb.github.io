import argparse
import wandb
import os
from architectures import sep_conv_net
from preprocess import dataGenerator, to_tf_dataset
from callbacks import model_checkpoint_callback, CustomCallback, reduce_lr
import tensorflow as tf
from wandb.keras import WandbCallback
import sys

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MODEL_NAME = 'facial_keypoints'
IMAGE_SIZE = (256,256)
OUTPUT_SIZE = 136

hyperparameter_defaults = dict(
    epochs = 20,
    loss = 'mse',
    batch_size = 32,
    steps_per_epoch = 100,
    validation_steps = 10,
    optimizer = "Adam",
    learning_rate = 1e-3,
    blocks = 3,
    dense_layers = 1,
    conv_per_block = 2,
    kernal_size = 3,
    activation = 'relu',
    batch_norm = 1,
    dropout = 0.5,
    depthwise_initializer = "glorot_uniform",
    pointwise_initializer = "glorot_uniform",
  )


def main(project_dir, dataset_dir, sweep):
  wandb.init(config=hyperparameter_defaults, project="facial_keypoints", dir=project_dir)
  config = wandb.config
  print('configs for this sweep:', config)

  train_generator = dataGenerator(f'{dataset_dir}/training_frames_keypoints.csv',
                                  f'{dataset_dir}/training',
                                  training=True,
                                  input_size=IMAGE_SIZE,
                                  batch_size=config.batch_size,
                                  shuffle=True)
  train_dataset = to_tf_dataset(train_generator,
                                batch_size=config.batch_size,
                                img_size=IMAGE_SIZE,
                                output_size=OUTPUT_SIZE)

  test_generator = dataGenerator( f'{dataset_dir}/test_frames_keypoints.csv',
                                  f'{dataset_dir}/test', training=False,
                                  input_size=IMAGE_SIZE,
                                  batch_size=config.batch_size,
                                  shuffle=True)
  test_dataset = to_tf_dataset( test_generator,
                                batch_size=config.batch_size,
                                img_size=IMAGE_SIZE,
                                output_size=OUTPUT_SIZE)

  model = sep_conv_net( input_shape=IMAGE_SIZE,
                        output_size=OUTPUT_SIZE,
                        blocks=config.blocks,
                        dense_layers=config.dense_layers,
                        conv_per_block=config.conv_per_block,
                        kernal_size=config.kernal_size,
                        activation=config.activation,
                        batch_norm=bool(config.batch_norm),
                        dropout=config.dropout,
                        depthwise_initializer=config.depthwise_initializer,
                        pointwise_initializer=config.pointwise_initializer)

  model.compile(loss=config.loss,
                optimizer=getattr(tf.keras.optimizers, config.optimizer)(learning_rate=config.learning_rate))

  # model.summary()
  
  model.fit(train_dataset, epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            validation_steps=config.validation_steps,
            validation_data=test_dataset,
            callbacks=[ WandbCallback(save_model=False),
                        reduce_lr,
                        *([model_checkpoint_callback(PROJECT_DIR, MODEL_NAME), CustomCallback(PROJECT_DIR, MODEL_NAME, IMAGE_SIZE)] if not sweep else [])
                      ])


if __name__ == '__main__':
  my_parser = argparse.ArgumentParser(description='Training script for facial keypoint CNN')

  my_parser.add_argument('--sweep',
                        '-s',
                        action='store_true',
                        help='flag if running as a sweep')

  args = my_parser.parse_args()

  os.chdir('/')

  IN_COLAB = os.path.isdir("content/drive/MyDrive")
  print('In Colab:', IN_COLAB)

  PROJECT_DIR = "content/drive/MyDrive/dylanlrrb.github.io/portfolio/facial_keypoints" if IN_COLAB else 'tf/notebooks/portfolio/facial_keypoints'
  DATASET_DIR = "content/drive/MyDrive/datasets/facial_keypoints" if IN_COLAB else 'tf/notebooks/portfolio/facial_keypoints/data'
  SWEEP = args.sweep

  main(PROJECT_DIR, DATASET_DIR, SWEEP)