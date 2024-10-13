######################################################################
# KanjiRec
# Helper tool to recognize Kanjis
# Copyright 2024, Andreas Gaiser
######################################################################
# Script for training Keras models
######################################################################

from tensorflow import keras
from keras import *
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import datetime
import common


image_scale = 30
label_count = 2999
if 1:

    training, validation = image_dataset_from_directory(
        common.TRANING_DATA_DIR,
        labels="inferred",
        # label_mode="categorical",
        color_mode="grayscale",
        batch_size=512,  # 256
        image_size=(image_scale, image_scale),
        validation_split=0.25,
        subset="both",
        seed=42,
    )

    class_names = training.class_names
    print(class_names)
    label_count = len(class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in training.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # print(images[i])
            plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
            print(labels[i])
            plt.title(str(labels[i]))
            plt.axis("off")
    plt.show()


RANDOM_SEED = 42


def build_model_A(image_x, image_y, label_count):
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(image_x, image_y, 1)),
            layers.RandomZoom(
                seed=RANDOM_SEED, height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)
            ),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.2, width_factor=0.2
            ),
            layers.Conv2D(2, 5, padding="same", activation="relu"),
            layers.Conv2D(4, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(8, 5, padding="same", activation="relu"),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(label_count // 16, activation="relu"),
            layers.Dense(label_count),
        ],
        name="ModelA",
    )
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model_B(image_x, image_y, label_count):
    model = Sequential(
        [
            layers.Rescaling(
                1.0 / 255, input_shape=(30, 30, 1)
            ),  # , input_shape=(None,None,1)),
            layers.RandomZoom(
                seed=RANDOM_SEED,
                height_factor=(-0.1, 0.2),
                width_factor=(-0.1, 0.2)
                # old: 0.0, 0.2 <=> 0.0, 0.2
            ),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.2, width_factor=0.2
            ),
            layers.Conv2D(6, 5, padding="same", activation="relu"),
            layers.Conv2D(4, 5, padding="same", activation="relu"),
            layers.Dropout(0.2),
            # layers.MaxPooling2D(),
            layers.Conv2D(label_count, 1, padding="same", activation="relu"),
            # layers.Conv2D(label_count // 16, kernel_size=(1, 1), activation="relu"),
            # layers.Conv2D(label_count, kernel_size=(1, 1), activation="relu"),
            layers.Flatten(),
        ],
        name="ModelB",
    )
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model_C(image_x, image_y, label_count):
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(image_x, image_y, 1)),
            layers.RandomZoom(
                seed=RANDOM_SEED, height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)
            ),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.2, width_factor=0.2
            ),
            layers.Conv2D(25, 15, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(50, 5, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(100, 2, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.Dense(label_count // 16, activation="relu"),
            # layers.Dense(label_count // 8, activation="relu"),
            layers.Dense(label_count),
        ],
        name="ModelC",
    )
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model_D(image_x, image_y, label_count):
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(image_x, image_y, 1)),
            layers.RandomZoom(
                seed=RANDOM_SEED, height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)
            ),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.2, width_factor=0.2
            ),
            layers.Conv2D(10, 15, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(9, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(8, 2, padding="same", activation="relu"),
            layers.Dropout(0.1),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(label_count // 32, activation="relu"),
            # layers.Dense(label_count // 8, activation="relu"),
            layers.Dense(label_count),
        ],
        name="ModelD",
    )
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_model_E(image_x, image_y, label_count):
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(image_x, image_y, 1)),
            layers.RandomZoom(
                seed=RANDOM_SEED, height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)
            ),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.2, width_factor=0.2
            ),
            layers.Conv2D(10, 15, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(10, 2, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(5, 2, padding="same", activation="relu"),
            layers.Dropout(0.1),
            layers.Flatten(),
            # layers.Dense(label_count // 64, activation="relu"),
            # layers.Dense(label_count // 64, activation="relu"),
            layers.Dense(label_count),
        ],
        name="ModelE",
    )
    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


if False:
    model = Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(image_x, image_y, 1)),
            layers.RandomZoom(seed=RANDOM_SEED, height_factor=0.1, width_factor=0.1),
            layers.RandomTranslation(
                seed=RANDOM_SEED, height_factor=0.1, width_factor=0.1
            ),
            layers.RandomContrast(seed=RANDOM_SEED, factor=0.05),
            layers.Conv2D(128, 5, padding="same", activation="relu"),
            layers.Conv2D(256, 5, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 4, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            # layers.Dense(label_count, activation="relu"),
            layers.Dense(label_count),
        ]
    )


model = build_model_E(image_scale, image_scale, label_count)
# model.summary()
from keras.saving import load_model

# model = load_model('.../models/2024-09-06_19_07_01.741946-ModelC_model_epoch_85_loaded.keras')
now = datetime.datetime.now()
PREFIX = ("%s-%s" % (now, model.name)).replace(":", "_").replace(" ", "_")
model.save("models/%s_model_initial.keras" % PREFIX)
for epoch in range(12500):
    model.fit(training, validation_data=validation, epochs=1)
    if epoch % 5 == 0:
        model.save("models/%s_model_epoch_%s_loaded.keras" % (PREFIX, epoch))
model.save("models/%s_model_epoch_final.keras" % PREFIX)
