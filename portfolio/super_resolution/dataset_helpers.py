import tensorflow as tf
import pathlib
import os
from functools import reduce

AUTOTUNE = tf.data.AUTOTUNE

def shard_dataset_splits(ds, train_ratio=8, valid_ratio=1, test_ratio=1):
  num_shards = train_ratio + valid_ratio + test_ratio
  shards = [ds.shard(num_shards=num_shards, index=i) for i in range(num_shards)]
  
  train_shards = shards[:train_ratio]
  valid_shards = shards[train_ratio:train_ratio+valid_ratio]
  test_shards  = shards[train_ratio+valid_ratio:]

  train_ds = reduce(lambda a, b: a.concatenate(b), train_shards)
  val_ds = reduce(lambda a, b: a.concatenate(b), valid_shards)
  test_ds  = reduce(lambda a, b: a.concatenate(b), test_shards)

  return train_ds, val_ds, test_ds


def split_training_image_dataset(
  dataset_name,
  dataset_url,
  train_transforms,
  val_transforms,
  test_transforms,
  train_val_test_ratio=(8,1,1),
  data_subsample=1,
  batch_size=128,
  image_size=(224,224),
  labels='inferred',
  shuffle=True,):

  data_dir = f'/root/.keras/datasets/{dataset_name}'

  # Do not re-download dadtaset if it already exists
  if not os.path.isdir(data_dir):
    dataset_url = f'https://datasets-349058029.s3.us-west-2.amazonaws.com/flickr/{dataset_name}.zip'
    extract = '.zip' in dataset_url or '.tar' in dataset_url
    tf.keras.utils.get_file(origin=dataset_url, extract=extract)

  data_dir = pathlib.Path(data_dir)
  print(f"{len(list(data_dir.glob('*/*.jpg')))} Images in Dataset")

  ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=labels,
    shuffle=shuffle,
    image_size=image_size,
    batch_size=batch_size,
    crop_to_aspect_ratio=True)
  
  ds = ds.take(int(data_subsample * len(ds)))
  print(f'dataset size: {len(ds)} batches of {batch_size} samples each')

  train_ratio, val_ratio, test_ratio = train_val_test_ratio

  train_ds, val_ds, test_ds = shard_dataset_splits(
    ds,
    train_ratio=train_ratio,
    valid_ratio=val_ratio,
    test_ratio=test_ratio,)

  train_ds = train_ds.map(train_transforms).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.map(val_transforms).prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.map(test_transforms).prefetch(buffer_size=AUTOTUNE)

  return train_ds, val_ds, test_ds

def iterate_dataset(dataset):
  iterator = iter(dataset)
  def generate_real():
    nonlocal iterator
    try:
      X_batch, y_batch = iterator.get_next()
    except tf.errors.OutOfRangeError:
      iterator = iter(dataset)
      X_batch, y_batch = iterator.get_next()
    return X_batch, y_batch
  return generate_real