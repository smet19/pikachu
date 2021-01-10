import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

from cnn_model import config
from cnn_model import __version__ as _version


def create_dataset(*, batch_size=32, random_seed=42):
    train_ds = image_dataset_from_directory(
        os.path.join(config.DATASET_DIR, 'train'),
        validation_split=0.2,
        subset="training",
        seed=random_seed,
        batch_size=batch_size)

    val_ds = image_dataset_from_directory(
        os.path.join(config.DATASET_DIR, 'train'),
        validation_split=0.2,
        subset="validation",
        seed=random_seed,
        batch_size=batch_size)

    test_ds = image_dataset_from_directory(
        os.path.join(config.DATASET_DIR, 'test'),
        seed=random_seed,
        batch_size=batch_size)

    return train_ds, val_ds, test_ds


def load_pipeline():
    pipe = load_model(f'{config.MODEL_PATH}_v{_version}')
    return pipe
