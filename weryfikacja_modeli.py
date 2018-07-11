
# coding: utf-8
import os
import Levenshtein
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
import keras.callbacks
import cv2
import itertools
import os
import glob
from os.path import join
from detekcja import predict

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g,', '--gpu', help='Which gpu to use', type=str, default="1")

args = vars(parser.parse_args())
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu']

def multiple_models():
    dirpath = '/home/yason/workspace/ocr/img/val'
    for file in glob.glob('logs/*/*'):
        if file.split('/')[2] == 'best_weights.hdf5':
            print("----------------------------------")
            print(file.split('/')[1])
            test = predict()
            test.load(file)
            test.collect_data(dirpath)
            test.decode()
            test.calculate_accuracy(False)

multiple_models()
