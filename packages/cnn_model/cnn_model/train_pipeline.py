from cnn_model import config
from cnn_model.data_management import create_dataset, load_pipeline
from cnn_model.pipeline import create_pipeline
from cnn_model import __version__ as _version


random_seed = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(random_seed)
 
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(random_seed)
 
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(random_seed)
 
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(random_seed)


def run_training(save_result: bool = True):
    train_ds, val_ds, _ = create_dataset()

    pipeline = create_pipeline()

    training = pipeline.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS
    )

    if save_result:
        pipeline.save(f'{config.MODEL_PATH}_v{_version}')


def run_eval():
    _, _, test_ds = create_dataset()
    pipe = load_pipeline()

    assert pipe is not None

    results = pipe.evaluate(test_ds)

    return results



if __name__ == '__main__':
    run_training(save_result=True)