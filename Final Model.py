# Importing libraries

import os
import cv2
import numpy as np
import kagglehub
import json
import random
import plotly.express as px
from collections import defaultdict
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

# ---------------------------------------------------------------------------------------

# Loading data.

kagglehub.login()  # Most to write your API 

path = kagglehub.competition_download('filament-segmentation-2026')
print("Path to competition files:", path)

def generate_mask_for_image(img_info, filaments):
    """
    It creates a binary mask (0 and 1).
    """
    height = img_info['height']
    width = img_info['width']

    # Background
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in filaments:
        segmentations = ann.get('segmentation', [])

        for seg in segmentations:
            poly = np.array(seg, dtype=np.int32).reshape((-1, 2))

            cv2.fillPoly(mask, [poly], color=1)

    return mask

# ---------------------------------------------------------------------------------------

# Creating data generator

class FilamentDataGenerator(Sequence):
    def __init__(self, images_list, annotations_map, img_dir, batch_size=8, target_size=(512, 512), shuffle=True):
        self.images_list = images_list
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images_list))
        self.on_epoch_end()

    def __len__(self):
      """
      Number of batch per epoch.
      """
      return int(np.ceil(len(self.images_list) / self.batch_size))

    def on_epoch_end(self):
      """
      Randomly shuffle the data for each epoch.
      """
      if self.shuffle:
          np.random.shuffle(self.indexes)

    def __getitem__(self, index):
      """
      Reads and resizes 8 photos and smoothes them.
      """

      batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

      batch_images = []
      batch_masks = []

      for idx in batch_indexes:
            img_info = self.images_list[idx]
            img_id = img_info['id']

            # Reading an image from disk.
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generating a binary mask of the same image.
            filaments = self.annotations_map.get(img_id, [])
            mask = generate_mask_for_image(img_info, filaments)

            # On-the-fly resizing.
            image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            # Data normalization (numbers between 0 and 1).
            image_normalized = image_resized.astype(np.float32)  # First try with (->{ \ 255.0 }<-)
            mask_normalized = np.expand_dims(mask_resized.astype(np.float32), axis=-1)

            batch_images.append(image_normalized)
            batch_masks.append(mask_normalized)

      return np.array(batch_images), np.array(batch_masks)

# ---------------------------------------------------------------------------------------

# Path

BASE_DIR = '/root/.cache/kagglehub/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
ANN_PATH = os.path.join(BASE_DIR, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')

# Reading json path

with open(ANN_PATH, 'r') as f:
    coco_data = json.load(f)
all_images = coco_data['images']

# ---------------------------------------------------------------------------------------

# Seperating data
train_images, val_images = train_test_split(all_images, test_size=0.25, random_state=19)
IMG_DIR = os.path.join(BASE_DIR, 'train', 'train_images')

filaments_by_image = defaultdict(list)
for ann in coco_data['annotations']:
    filaments_by_image[ann['image_id']].append(ann)

train_generator = FilamentDataGenerator(
    images_list=train_images,
    annotations_map=filaments_by_image,
    img_dir=IMG_DIR,
    batch_size=8,
    target_size=(512, 512),
    shuffle=True
)

val_generator = FilamentDataGenerator(
    images_list=val_images,
    annotations_map=filaments_by_image,
    img_dir=IMG_DIR,
    batch_size=8,
    target_size=(512, 512),
    shuffle=False
)

print(f"Number of batch Train: {len(train_generator)}")
print(f"Number of batch Validation: {len(val_generator)}")

# ---------------------------------------------------------------------------------------

# Importing Model and creating handmade blocks

encoder = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(512, 512, 3)
)

encoder.trainable = True

for layer in encoder.layers[:-40]:
    layer.trainable = False

skip_names = [
    "block2a_expand_activation",
    "block3a_expand_activation",
    "block4a_expand_activation",
    "block6a_expand_activation"
]

skip_outputs = [
    encoder.get_layer(name).output
    for name in skip_names
]

# ---------------------------------------------------------------------------------------

# Importing Model and encoder part

encoder_model = Model(
    inputs=encoder.input,
    outputs=skip_outputs
)

# Creating handmade decoder part

def decoder_block(x, skip, filters):

    x = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(x)

    x = layers.Concatenate()([x, skip])

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    return x

# ---------------------------------------------------------------------------------------

# Creating main layers and conecting them to Model and handmade decoder/encoder

def build_transfer_unet():

    inputs = layers.Input(shape=(512, 512, 3))

    skip1, skip2, skip3, skip4 = encoder_model(inputs)

    x = skip4

    x = decoder_block(x, skip3, 256)

    x = decoder_block(x, skip2, 128)

    x = decoder_block(x, skip1, 64)

    x = layers.Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same"
    )(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid"
    )(x)

    return Model(inputs, outputs)

# ---------------------------------------------------------------------------------------

# Creating Dice part

def dice_coef(y_true, y_pred, smooth=1e-5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coef(y_true, y_pred)
    return 0.3 * bce + 0.7 * dice

transfer_model = build_transfer_unet()

print(transfer_model.output_shape)

# ---------------------------------------------------------------------------------------

# Compiling and Fiting Fine-tune model
transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4  # Learning rate must to be high; otherwise model will underfit
    ),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

callbacks = [
    ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, min_lr=1e-5, mode='max', verbose=1),
    ModelCheckpoint('best_filament_model.keras', monitor='val_dice_coef', save_best_only=True, mode='max', verbose=1)
]

history_transfer = transfer_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# ---------------------------------------------------------------------------------------

# Drowing plot
print(history_transfer.history.keys())

epochs = list(range(1, len(history_transfer.history['loss']) + 1))

fig_loss = px.line(
    x=epochs,
    y=[history_transfer.history['loss'], history_transfer.history['val_loss']],
    labels={'x': 'Epoch', 'value': 'Loss', 'variable': 'Dataset'},
    title='Loss Curve',
    markers=True
)
fig_loss.show()
fig_dice = px.line(
    x=epochs,
    y=[history_transfer.history['dice_coef'], history_transfer.history['val_dice_coef']],
    labels={'x': 'Epoch', 'value': 'Dice Score', 'variable': 'Dataset'},
    title='Dice Score',
    markers=True
)
fig_dice.show()

# ---------------------------------------------------------------------------------------

# Choosing samples and predicting them by Model
sample_val_images = random.sample(val_images, 200)

sample = FilamentDataGenerator(
  images_list=sample_val_images,
  annotations_map=filaments_by_image,
  img_dir=IMG_DIR,
  batch_size=200,
  target_size=(512, 512),
  shuffle=True
)

x_batch, y_true = sample[0]

predictions = transfer_model.predict(x_batch, batch_size=10)

y_pred = (predictions > 0.45).astype(int)
y_true = (y_true > 0.45).astype(int)

"""
Various threshold levels were tested, and 0.45 was determined to be the optimal value.
"""

def dice_metric(y_true, y_pred, smooth=2e-4):  # Learning rate must to be high; otherwise model will underfit
  """
  Dice = (2 x |Y ∩ Ŷ|) ÷ (|Y| + |Ŷ|) = (2 x TP) ÷ (2 x TP + FP + FN)
  """
  intersection = np.sum(y_true * y_pred)  # |Y ∩ Ŷ|
  total = np.sum(y_true) + np.sum(y_pred)  # |Y| + |Ŷ|

  return (2.0 * intersection + smooth) / (total + smooth)

print(f"Dice metric result: {(dice_metric(y_true, y_pred) * 100):.2f}")
