import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
import sys
import pathlib
import os

hyperparameter_tuning = len(sys.argv) < 2 or len(sys.argv) > 3

print('hyperparameter tuning', hyperparameter_tuning)
if not hyperparameter_tuning:
        print(f'with {sys.argv[1]} fc layers and {sys.argv[2]} nodes per layer')

weight_map = {}
weights = [1,1]

for idx, value in enumerate(weights):
  weight_map[idx] = value

dataset_name = 'cat_vs_dog'
data_dir = f'/root/.keras/datasets/{dataset_name}'

if not os.path.isdir(data_dir):
        dataset_url = f'https://datasets-349058029.s3.us-west-2.amazonaws.com/cat_dog_classifier/{dataset_name}.zip'
        tf.keras.utils.get_file(origin=dataset_url, extract=True)

data_dir = pathlib.Path(data_dir)
print(f"{len(list(data_dir.glob('*/*.jpg')))} images in dataset")

batch_size = 128
img_size = 224

ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  image_size=(img_size, img_size),
  batch_size=batch_size)

num_classes = len(ds.class_names)

if hyperparameter_tuning:
        ds = ds.take(int(0.1 * len(ds)))

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

data_augmentation_fn = lambda x, y: (data_augmentation(x, training=True), y)

AUTOTUNE = tf.data.AUTOTUNE

train_split = 0.8
val_split = 0.1

train_size = int(train_split * len(ds))
val_size = int(val_split * len(ds))

train_ds = ds.take(train_size).map(data_augmentation_fn, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE) 
val_ds = ds.skip(train_size).take(val_size).prefetch(buffer_size=AUTOTUNE)
test_ds = ds.skip(train_size).skip(val_size).prefetch(buffer_size=AUTOTUNE) 

if hyperparameter_tuning:
        hidden_layer_nums = [1,2,3]
        hidden_layer_node_nums = [100, 200, 500, 700]
else:
        hidden_layer_nums = [int(sys.argv[1])]
        hidden_layer_node_nums = [int(sys.argv[2])]

epochs = 15

for hidden_layer_num in hidden_layer_nums:
        for hidden_layer_node_num in hidden_layer_node_nums:
                tf.keras.backend.clear_session()
                model_name = f'{dataset_name}_{hidden_layer_num}layers_{hidden_layer_node_num}nodes'
                hidden_layers = []
                for i in range(hidden_layer_num):
                        hidden_layers.append(layers.Dense(hidden_layer_node_num, activation = 'relu'))
                        hidden_layers.append(layers.Dropout(0.2))

                feature_extractor = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                                        include_top=False,
                                                        weights='imagenet')
                feature_extractor.trainable = False

                model = tf.keras.Sequential([
                        layers.InputLayer(input_shape=(img_size, img_size, 3)),
                        layers.Rescaling(1./127.5, offset=-1),
                        feature_extractor,
                        layers.GlobalAveragePooling2D(),
                        *hidden_layers,
                        layers.Dense(num_classes, activation = None)])

                model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

                model.summary()

                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=f'logs/{dataset_name}/{model_name}',
                        histogram_freq=1)

                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=f'models/{model_name}.h5',
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

                history = model.fit(
                        train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        class_weight=weight_map,
                        callbacks=[*([tensorboard_callback] if hyperparameter_tuning else [model_checkpoint_callback])])
                
                if not hyperparameter_tuning:
                        print('running best model on test set:')
                        tf.keras.backend.clear_session()
                        model = tf.keras.models.load_model(f'models/{model_name}.h5')
                        model.evaluate(test_ds)
