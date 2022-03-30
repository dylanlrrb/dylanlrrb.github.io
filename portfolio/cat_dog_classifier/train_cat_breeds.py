import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
import sys
import pathlib
import os

class_freq = [459,992,150,5296,20,136,255,2678,428,2036,768,345,9,3469,5,85,25,4,172,18,120,3231,3153,4500,5483,10180,506,472,185,1302,128,26,69,18,1619,2060,182,149,581,119,38,492,102,4219,112,135,2870,2071,381,78,3089,190,95,23,1626,37,410,3013,2257,261,3397,3964,751,816,3182,2,]

hyperparameter_tuning = len(sys.argv) < 2 or len(sys.argv) > 3

print('hyperparameter tuning', hyperparameter_tuning)
if not hyperparameter_tuning:
        print(f'with {sys.argv[1]} fc layers and {sys.argv[2]} nodes per layer')

dataset_name = 'cat_breeds'
data_dir = f'/root/.keras/datasets/{dataset_name}'

if not os.path.isdir(data_dir):
        dataset_url = f'https://datasets-349058029.s3.us-west-2.amazonaws.com/cat_dog_classifier/{dataset_name}.zip'
        tf.keras.utils.get_file(origin=dataset_url, extract=True)

data_dir = pathlib.Path(data_dir)
n_samples = len(list(data_dir.glob('*/*.jpg')))
print(f"{n_samples} images in dataset")

batch_size = 128
img_size = 224

ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  image_size=(img_size, img_size),
  batch_size=batch_size)

num_classes = len(ds.class_names)

class_weights = [n_samples / (num_classes * freq) if freq > 0 else 1 for freq in class_freq]
weight_map = dict(enumerate(class_weights))

if hyperparameter_tuning:
        ds = ds.take(int(0.03 * len(ds)))

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
        hidden_layer_nums = [1,2]
        hidden_layer_node_nums = [500, 1000, 2000, 2500, 3000]
else:
        hidden_layer_nums = [int(sys.argv[1])]
        hidden_layer_node_nums = [int(sys.argv[2])]

epochs = 10

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
                        layers.Dense(num_classes, activation = 'softmax')])

                model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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
