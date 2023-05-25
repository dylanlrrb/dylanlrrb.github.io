import os
import io
import gc
import cv2
import numpy as np
import pickle as pkl
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from keras import backend as K
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
from pycocotools.coco import COCO
from architectures import u_net
from dataset.preprocess import cocoDataGenerator, CategoryMappingHelper, to_tf_dataset
from dataset.augmentation import augment
import wandb
from wandb.keras import WandbCallback

# 2 = Warn messages and below are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.chdir('/')
PROJECT_DIR = 'tf/notebooks/portfolio/image_segmentation'
DATASET_DIR = 'root/datasets/coco2017'

MODEL_NAME = 'mobile_unet'
# MODEL_NAME = 'attn_mobile_unet'
# MODEL_NAME = 'tanh_attn_mobile_unet'
BATCH_SIZE = 4
IMAGE_SIZE = (224,224)
LR = 1e-3

train_annotations = COCO(f'{DATASET_DIR}/annotations/instances_train2017.json')
val_annotations = COCO(f'{DATASET_DIR}/annotations/instances_val2017.json')

catIDs = train_annotations.getCatIds()
cats = train_annotations.loadCats(catIDs)
num_classes = len(cats) + 1 # add one for the background class

catMapper = CategoryMappingHelper(train_annotations)

val_generator = cocoDataGenerator(val_annotations, f'{DATASET_DIR}/images/val2017', catMapper, input_size=IMAGE_SIZE, output_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = to_tf_dataset(val_generator, BATCH_SIZE, IMAGE_SIZE, num_classes)

train_generator = cocoDataGenerator(train_annotations, f'{DATASET_DIR}/images/train2017', catMapper, input_size=IMAGE_SIZE, output_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, train=True)
train_augment = dict(featurewise_center = False, samplewise_center = False,
                rotation_range = 10, width_shift_range = 0.01,
                height_shift_range = 0.01, brightness_range = (0.8,1.2),
                shear_range = 0.01, zoom_range = [1, 1.25],
                horizontal_flip = True, vertical_flip = False,
                fill_mode = 'reflect', data_format = 'channels_last')
train_aug_generator = augment(train_generator, train_augment)
train_dataset = to_tf_dataset(train_aug_generator, BATCH_SIZE, IMAGE_SIZE, num_classes)

model_checkpoint_callback = ModelCheckpoint(
        filepath=f'{PROJECT_DIR}/models/{MODEL_NAME}.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    # min_delta=0.001,
    factor=0.1,   
    patience=10, 
    min_lr=0,)

samples = [ {'name': 'dog', 'extension': 'jpg', 'description': 'Dog'},
            {'name': 'frisbee', 'extension': 'jpg', 'description': 'Person with Frisbee'},
            {'name': 'tennis', 'extension': 'jpg', 'description': 'Person with Racket'},
            {'name': 'hot_dog', 'extension': 'jpg', 'description': 'People with Hotdogs'},
            {'name': 'cat', 'extension': 'jpg', 'description': 'Cat with Computer'},
            {'name': 'ducks', 'extension': 'jpg', 'description': 'Ducks'},
        ]

def create_sparse_masks(model, image):
    X = np.array([cv2.resize(image, IMAGE_SIZE)])
    y = model.predict(X)
    y = np.squeeze(y)
    return np.argmax(y, axis=2)

def log_masks_to_wandb(sparse_masks, image, description, class_labels=None):
    image = np.array([cv2.resize(image, IMAGE_SIZE)])
    mask_data = np.array(sparse_masks)
    mask_img = wandb.Image(image, masks={
        "predictions": {
            "mask_data": mask_data,
            "class_labels": catMapper.filter_idx_to_category_name if class_labels==None else class_labels,
        },
    })
    wandb.log({description: mask_img})

def create_mask_plot(epoch, sparse_masks, image):
    X = np.array([cv2.resize(image, IMAGE_SIZE)])
    cat_pixel_counts = np.bincount(sparse_masks.flatten())
    above_zero = np.count_nonzero(cat_pixel_counts > 0)
    top_n = above_zero if above_zero < 5 else 5
    most_freq_idxs = np.argpartition(cat_pixel_counts, -top_n)[-top_n:]
    class_labels = {int(filter_idx): catMapper.get(filter_idx, 'filter_idx', 'category_name') for filter_idx in most_freq_idxs.astype(int)}
    most_freq_idxs = [*filter(lambda x: x != 80, most_freq_idxs.tolist())] # filter out the background class
    masks = np.zeros(IMAGE_SIZE)
    cmap = cm.viridis
    figure, (ax1, ax2) = plt.subplots(1, 2)
    
    for i, filter_idx in enumerate(most_freq_idxs[::-1]):
        pixel_value = len(most_freq_idxs)-i
        masks = np.maximum((sparse_masks == filter_idx)*pixel_value, masks)
        mycolor = cmap(pixel_value / (len(most_freq_idxs)))
        plt.plot(0, 0, "o", color=mycolor, label=f"{catMapper.get(filter_idx, 'filter_idx', 'category_name')}")
    plt.plot(0, 0, "o", color=cmap(0), label="background")

    ax1.text(0., 1., f'{epoch}'.zfill(2),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', edgecolor='none'))
    ax1.imshow(X.squeeze(), cmap=cmap)
    ax1.axis('off')
    ax2.imshow(masks, cmap=plt.cm.viridis)
    ax2.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.legend(loc="upper right", bbox_to_anchor=(1.6, 1.))
    
    return figure, class_labels

def create_attention_heatmap_plot(model, image, epoch):
    X = np.array([cv2.resize(image, IMAGE_SIZE)])
    a1, a2, a3, a4 = model.predict(X)

    figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.text(0., 1., f'{epoch}'.zfill(2),
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes,
        fontsize=15,
        bbox=dict(facecolor='white', edgecolor='none'))
    ax1.imshow(X.squeeze())
    ax1.axis('off')
    ax2.imshow(np.squeeze(a4), cmap='rainbow')
    ax2.axis('off')
    ax3.imshow(np.squeeze(a3), cmap='rainbow')
    ax3.axis('off')
    ax4.imshow(np.squeeze(a2), cmap='rainbow')
    ax4.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return figure

def plot_to_img(figure):
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  return image

class CustomCallback(Callback):
    def __init__(self, model_name, attention_model=None):
        self.model_name = model_name
        self.attention_model = attention_model
        super().__init__()
    def on_epoch_end(self, epoch, logs=None):
        for sample in samples:
            image = np.asarray(Image.open(f'{PROJECT_DIR}/assets/{sample["name"]}.{sample["extension"]}'))
            sparse_masks = create_sparse_masks(self.model, image)
            mask_plot, class_labels = create_mask_plot(epoch, sparse_masks, image)
            mask_img = plot_to_img(mask_plot)
            if self.attention_model:
                attention_heatmap_plot = create_attention_heatmap_plot(self.attention_model, image, epoch)
                attention_heatmap_img = plot_to_img(attention_heatmap_plot)
                wandb.log({f"{sample['description']} Attention Heatmap": wandb.Image(attention_heatmap_img)})
            wandb.log({sample['description']: wandb.Image(mask_img)})
            log_masks_to_wandb(sparse_masks, image, f"{sample['description']} Overlay", class_labels=class_labels)
        gc.collect()

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def tversky_index(y_true, y_pred, alpha=0.7):
    ones = tf.ones(K.shape(y_true))
    p0 = y_pred      # prob that pixels are class i
    p1 = ones-y_pred # prob that pixels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + (1-alpha)*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range 0-Ncl
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return T/Ncl

def focal_tversky_loss(alpha=0.7, gamma=0.75):
    def loss(y_true, y_pred):
        tvi = tversky_index(y_true, y_pred, alpha=alpha)
        return K.pow((1 - tvi), gamma)
    return loss


if __name__ == "__main__":

    conv_per_block = 1
    epochs = 100
    steps_per_epoch = 100 if BATCH_SIZE >= 4 else 200
    validation_steps = 10
    alpha = 0.7
    gamma = 0.75
    final_activation = 'softmax'

    wandb.init(project="image-segmentation", dir=PROJECT_DIR, name=MODEL_NAME, config={
        'conv_per_block': conv_per_block,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "alpha": alpha,
        "gamma": gamma,
        "num_classes": num_classes,
        "final_activation": final_activation,
    })

    model = u_net.define_mobile_unet(input_shape=IMAGE_SIZE, conv_per_block=conv_per_block, num_classes=num_classes, final_activation=final_activation)
    # model, attention_model = u_net.define_attention_mobile_unet(input_shape=IMAGE_SIZE, conv_per_block=conv_per_block, num_classes=num_classes, final_activation=final_activation)
    # model = u_net.define_vgg_unet(input_shape=IMAGE_SIZE, conv_per_block=conv_per_block, num_classes=num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                    loss=focal_tversky_loss(alpha=alpha, gamma=gamma),
                    metrics=['accuracy', sensitivity, specificity])
    
    # attention_model.compile()

    model.summary()

    model_history = model.fit(train_dataset, epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            validation_data=val_dataset,
                            callbacks=[reduce_lr,
                                model_checkpoint_callback,
                                WandbCallback(save_model=False),
                                CustomCallback(MODEL_NAME)])
                                # CustomCallback(MODEL_NAME, attention_model)])
