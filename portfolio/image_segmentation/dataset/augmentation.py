from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import gc

def augment(data_gen, imageDataGeneratorArgs={}, seed=None):
  rng = np.random.default_rng(seed if seed is not None else np.random.choice(range(9999)))

  X_gen = ImageDataGenerator(**imageDataGeneratorArgs)
  imageDataGeneratorArgs_mask = imageDataGeneratorArgs.copy()
   # Remove the brightness argument for the binary masks but keep spatial augmentations.
  imageDataGeneratorArgs_mask.pop('brightness_range', None)
  y_gen = ImageDataGenerator(**imageDataGeneratorArgs_mask)
  
  for X_batch, y_batch in data_gen:
    seed = rng.choice(range(9999))
    g_x = X_gen.flow(X_batch, 
                    batch_size = X_batch.shape[0], 
                    seed = seed, 
                    shuffle=False)
    g_y = y_gen.flow(y_batch, 
                    batch_size = y_batch.shape[0], 
                    seed = seed, 
                    shuffle=False)
      
    X_aug = next(g_x)
    y_aug = next(g_y)

    yield X_aug, y_aug
