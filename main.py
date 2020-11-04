import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from functions import *
import winsound

tf.enable_eager_execution()

content_path = 'images/input_images/' + input("Please type the input file (e.g. image.jpg): ")
while not Path(content_path).is_file():
    print("Image not found.")
    content_path = 'images/input_images/' + input("Please type the input file (e.g. image.jpg): ")

style_path = 'images/paintings/' + input("Please type the painting (e.g. painting.jpg): ")
while not Path(style_path).is_file():
    print("Image not found.")
    style_path = 'images/paintings/' + input("Please type the painting (e.g. painting.jpg): ")

best, best_loss = run_style_transfer(content_path,
                                     style_path, num_iterations=1000)

Image.fromarray(best)

show_results(best, content_path, style_path)
for i in range(3):
    winsound.Beep(1000, 200)