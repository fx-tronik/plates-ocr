
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
from shutil import copyfile


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g,', '--gpu', help='Which gpu to use', type=str, default="1")

args = vars(parser.parse_args())
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu']

def multiple_models():
    best_acc = 0
    dirspath = ('img/val5',)
    for file in glob.glob('logs/*/*'):
        if file.split('/')[2] == 'best_weights.hdf5':
            print("----------------------------------")
            print(file.split('/')[1])
            test = predict()
            test.collect_data(dirspath)
            test.load(file)
            test.decode()
            acc = test.calculate_accuracy(False)
            if acc > best_acc:
                best_model = file
                best_acc = acc
            keras.backend.clear_session()
    print('najlepszy model dla danego zbioru: ' + best_model)
    print('accuracy: ' + str(best_acc))
    copyfile(best_model, 'best_weights.hdf5')

multiple_models()
