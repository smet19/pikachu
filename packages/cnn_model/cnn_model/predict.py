import tensorflow as tf
import numpy as np

from cnn_model import __version__ as _version
from cnn_model import config
from cnn_model.data_management import load_pipeline


def make_prediction(*, img_url, img_id):
    img_path = tf.keras.utils.get_file(f'{img_id}', origin=img_url)
    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(config.IMG_SIZE[0], config.IMG_SIZE[1])
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    pipe = load_pipeline()

    prediction = pipe.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    return dict(prediction=config.CLASS_NAMES[np.argmax(score)],
                score=np.max(score),
                version=_version)


