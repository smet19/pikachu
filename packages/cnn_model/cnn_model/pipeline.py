import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from cnn_model import config


def create_pipeline():
    data_augmentation = Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                         input_shape=(config.IMG_SIZE[0], 
                                                                      config.IMG_SIZE[1],
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        layers.experimental.preprocessing.Resizing(config.IMG_SIZE[0], config.IMG_SIZE[1], interpolation='bilinear', name=None),
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3), name=None),
        layers.Conv2D(16, 3, padding='same', activation='relu', name=None),
        layers.MaxPooling2D(name=None),
        layers.Conv2D(32, 3, padding='same', activation='relu', name=None),
        layers.MaxPooling2D(name=None),
        layers.Conv2D(64, 3, padding='same', activation='relu', name=None),
        layers.MaxPooling2D(name=None),
        layers.Dropout(0.2),
        layers.Flatten(name=None),
        layers.Dense(128, activation='relu', name=None),
        layers.Dense(config.NUM_CLASSES, name=None)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


