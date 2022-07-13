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

wandb.init(config=hyperparameter_defaults, project="facial_keypoints_sweep")
config = wandb.config


def main():
  wandb.log({'loss': random.randint(0,9)})


if __name__ == '__main__':
  main()