import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--file", dest="filename", help="File for prediction", metavar="FILE"
)
parser.add_argument("-a", "--all", help="Show all images", action="store_true")
parser.add_argument("-g", "--gpu", help="Run on gpu", action="store_true")
args = parser.parse_args()

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
import books

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model/mask_rcnn_books_0010.h5")
# Download COCO trained weights from Releases if needed

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(books.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
if args.gpu:
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
else:
    with tf.device("/cpu:0"):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ["BG", "book"]


def visualizeImage(image):
    # Run detection
    results = model.detect([image], verbose=0)

    # Visualize results
    r = results[0]
    visualize.display_instances(
        image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"]
    )


def showImage(filename):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
    visualizeImage(image)


def showAllImages():
    file_names = next(os.walk(IMAGE_DIR))[2]
    for file_name in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
        visualizeImage(image)


def showRandomImage():
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    visualizeImage(image)


if args.filename:
    showImage(args.filename)
elif args.all:
    showAllImages()
else:
    showRandomImage()

