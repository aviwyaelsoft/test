# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 07:35:18 2017

@author: yael
"""

import os
import numpy as np
np.random.seed(123)
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt


import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, UpSampling2D
from keras.utils import np_utils

from skimage.io import imread
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th')

