import os
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from train import train_dataset, val_dataset, sensitivity, specificity, tversky_loss, CustomCallback

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_DIR = '../tf/notebooks/portfolio/image_segmentation'

BASE_MODEL_NAME = 'u_sep_1'
MODEL_NAME = 'u_sep_1_fine'
BATCH_SIZE = 4
IMAGE_SIZE = (224,224)
LR = 1e-10


tensorboard_callback = TensorBoard(
                        log_dir=f'{PROJECT_DIR}/logs/{MODEL_NAME}')

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




model = tf.keras.models.load_model(f'{PROJECT_DIR}/models/{BASE_MODEL_NAME}.h5', compile=False)
for layer in model.layers:
  if not isinstance(layer, BatchNormalization):
    layer.trainable = True

# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                loss=tversky_loss(),
                metrics=['accuracy', sensitivity, specificity],)

model_history = model.fit(train_dataset, epochs=50,
                          steps_per_epoch=100,
                          validation_steps=10,
                          validation_data=val_dataset,
                          callbacks=[reduce_lr,
                            tensorboard_callback,
                            model_checkpoint_callback,
                            CustomCallback(MODEL_NAME)])