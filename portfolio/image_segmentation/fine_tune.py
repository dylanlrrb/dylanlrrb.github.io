import os
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from train import train_dataset, val_dataset, sensitivity, specificity, focal_tversky_loss, CustomCallback
import wandb
from wandb.keras import WandbCallback

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_DIR = '../tf/notebooks/portfolio/image_segmentation'
ABS_PROJECT_DIR = 'tf/notebooks/portfolio/image_segmentation'

BASE_MODEL_NAME = 'mobilnet_u_sep_1_tvl-gamma'
MODEL_NAME = 'finetune_mobilnet_u_sep_1_tvl-gamma'
BATCH_SIZE = 4
IMAGE_SIZE = (224,224)
LR = 1e-10
alpha = 0.7
gamma = 0.75
epochs = 50
steps_per_epoch = 100
validation_steps = 10

model_checkpoint_callback = ModelCheckpoint(
        filepath=f'{PROJECT_DIR}/models/{MODEL_NAME}.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,   
    patience=10, 
    min_lr=0,)


wandb.init(project="image-segmentation", dir=ABS_PROJECT_DIR, name=MODEL_NAME, config={
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "alpha": alpha,
        "gamma": gamma,
    })


model = tf.keras.models.load_model(f'{PROJECT_DIR}/models/{BASE_MODEL_NAME}.h5', compile=False)
# unfreeze backbone model
for layer in model.layers:
  if not isinstance(layer, BatchNormalization):
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss=focal_tversky_loss(alpha=alpha, gamma=gamma),
                metrics=['accuracy', sensitivity, specificity],)

model_history = model.fit(train_dataset, epochs=epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          validation_data=val_dataset,
                          callbacks=[reduce_lr,
                            model_checkpoint_callback,
                            WandbCallback(save_model=False),
                            CustomCallback(MODEL_NAME)])