import os


PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD))

DATASET_DIR = os.path.join(PACKAGE_ROOT, 'dataset')
MODEL_PATH = os.path.join(PACKAGE_ROOT, 'saved_models/cnn_model')

IMG_SIZE = (200, 200)
NUM_CLASSES = 2
EPOCHS = 10
CLASS_NAMES = ["pikachu", "notpikachu"]

