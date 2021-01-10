import tensorflow as tf

from cnn_model.data_management import create_dataset


def test_create_dataset_return_datasets():
    train_ds, val_ds, test_ds = create_dataset()

    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

