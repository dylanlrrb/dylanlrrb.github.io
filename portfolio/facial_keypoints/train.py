import argparse
import re
import wandb
import random





hyperparameter_defaults = dict(
    dropout = 0.5,
    channels_one = 16,
    channels_two = 32,
    batch_size = 100,
    learning_rate = 0.001,
    epochs = 2,
    )


def main(project_dir, dataset_dir):
  wandb.init(config=hyperparameter_defaults, project="facial_keypoints_sweep", dir=PROJECT_DIR)
  config = wandb.config


  wandb.log({'loss': random.randint(0,9)})
  print(project_dir, dataset_dir)


if __name__ == '__main__':
  # my_parser = argparse.ArgumentParser(description='Training script for facial keypoint CNN')

  # my_parser.add_argument('--project_dir',
  #                       '-p',
  #                       type=str,
  #                       required=False,
  #                       help='path to the project relitive to /')
  # my_parser.add_argument('--dataset_dir',
  #                       '-d',
  #                       '--dataset_dir',
  #                       type=str,
  #                       required=False,
  #                       help='path to the dataset relitive to /')

  # args = my_parser.parse_args()

  # PROJECT_DIR = args.project_dir if args.project_dir != None else 'tf/notebooks/portfolio/facial_keypoints'
  # DATASET_DIR = args.dataset_dir if args.dataset_dir != None else 'tf/notebooks/portfolio/facial_keypoints/data'

  PROJECT_DIR = 'tf/notebooks/portfolio/facial_keypoints'
  DATASET_DIR = 'tf/notebooks/portfolio/facial_keypoints/data'

  main(PROJECT_DIR, DATASET_DIR)