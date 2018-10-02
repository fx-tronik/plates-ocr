# coding: utf-8
import os


import keras
import tensorflow as tf
print("-----------------------")
print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

from os.path import join
import pprint
import json
import random
import itertools
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Bidirectional, BatchNormalization
from keras.layers import Reshape, Lambda, CuDNNGRU, Dropout, SpatialDropout1D
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, Adam, Nadam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.utils import multi_gpu_model
import keras.callbacks
import cv2
from keras.preprocessing.image import ImageDataGenerator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dirpath', help='savepath for model')
parser.add_argument('-o', '--optimizer', help='optimizer (adam, nadam)', default='adam')
parser.add_argument('-g,', '--gpu', help='Which gpu to use', type=str, default="0")
parser.add_argument('-dr', '--dropout', type=float, default=0.0)
parser.add_argument('-r', '--rnn', type=str,  help='gru or cudnn')
args = vars(parser.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu']
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.FATAL)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

from collections import Counter
def get_counter(dirpath):
    dirname =  os.path.basename(dirpath)
    letters = ''
    lens = []
    for filename in os.listdir(dirpath):
        if filename[-3:] == "jpg":
            filename = filename.split('.',1)[0]
            description = filename
            lens.append(len(description))
            letters += description
    print("-----------------------")
    print('Max plate length in "%s":' % dirname, max(Counter(lens).keys()))
    return {'counter':Counter(letters), 'amount':max(Counter(lens).keys())}


collection = get_counter('img/train2')
letters = sorted([' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z', '1', '2', '3',
           '4', '5', '6', '7', '8', '9', '0'])
print("-----------------------")
# letters_train = set(collection['counter'].keys())
# letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))
print("-----------------------")
def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

class TextImageGenerator:

    def __init__(self, folders, img_w, img_h, batch_size, downsample_factor, val, max_text_len=8, synt=0):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.synt_amount = 0
        self.samples = []

        for dirpath in folders:

            for filename in os.listdir(dirpath):
                img_filepath = join(dirpath, filename)
                if filename[-3:] == "jpg":
                    if filename.startswith(('f_', 't_')):
                        filename = filename.split('_', 1)[1]
                    if filename[4] == '_':
                        filename = filename.split('_', 1)[1]
                    filename = filename.split('.', 1)[0]
                    self.samples.append([img_filepath, filename])

        if val:
            print("-----------------------")
            print ("[INFO] %d probek w zbiorze walidacyjnym" % len(self.samples) )
        else:
            print("-----------------------")
            print ("[INFO] %d probek w zbiorze uczącym" % len(self.samples) )

        if (synt != 0):
            syntetic = os.listdir('img/synt/')
            random.shuffle(syntetic)
            syntetic = syntetic[0:synt]
            for filename in syntetic:
                img_filepath = join('img/synt/', filename)
                if filename[-3:] == "jpg":
                    filename = filename.split('.',1)[0]
                    # print(filename)
                    # print(img_filepath)
                    self.samples.append([img_filepath, filename])
            print("[INFO] Dodano %d syntetycznych probek" % len(syntetic) )
            self.synt_amount = len(syntetic)



        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.current_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def build_data(self):
        print("[INFO] Wczytywanie danych")
        for i, (img_filepath, text) in enumerate(self.samples):
            #print(img_filepath)
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            self.imgs[i, :, :] = img
            self.texts.append(text)

        #datagen.standardize(self.imgs)

    def get_output_size(self):
        return len(letters) + 1

    def next_sample(self):
        self.current_index += 1
        if self.current_index >= self.n:
            self.current_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.current_index]], self.texts[self.indexes[self.current_index]]

    def next_batch(self):
        print("[INFO]Next Batch")
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
                print('channel_first')
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)

                X_data[i] = img
                Y_data[i][:len(text_to_labels(text))] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)

            self.inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (self.inputs, outputs)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def optimizer(x):

    switcher = {
        'adam': lambda: Adam(lr=0.0001),
        'nadam': lambda: Nadam(lr=0.002),
        'sgd' :  lambda: SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
    }

    method = switcher.get(x)
    return method()

def train(p_batch_size, p_dropout1, p_dropout2, p_dir, p_optimizer, p_rnn_size, p_synt, rnn, norm):
    sess = tf.Session(config = config)

    K.set_session(sess)

    # Input Parameters
    img_h = 60
    img_w = 260
    # Network parameters
    conv_filters = 16
    conv_filters2 = conv_filters * 2
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = p_rnn_size
    synt = p_synt

    folders = ('img/train2', 'img/train3', 'img/train4', 'img/train5')
    val_folders = ('img/val5',)

    dir = p_dir
    working_dir = 'logs/' + dir
    filepath = working_dir + '/best_weights.hdf5'

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    tb = keras.callbacks.TensorBoard(log_dir = working_dir)

    if rnn=='cudnn':
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=25, verbose=1, mode='min')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=True)
    else:
        earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1, mode='min')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=True)


    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    batch_size = p_batch_size
    downsample_factor = pool_size ** 2
    tiger_train = TextImageGenerator(folders, 260, 60, batch_size, 4,False, collection['amount'], synt=synt)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(val_folders, 260, 60, batch_size, 4,True, collection['amount'])
    tiger_val.build_data()

    print("-----------------------")
    print("[INFO] Budowanie modelu")
    print("-----------------------")
    act = 'relu'

    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    if norm:
        inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1_1')(inner)
    if norm:
       inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                  activation=act, kernel_initializer='he_normal',
                  name='conv1_2')(inner)
    if norm:
        inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                activation=act, kernel_initializer='he_normal',
                name='conv1_3')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters2, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    if norm:
        inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters2, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2_1')(inner)
    if norm:
        inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters2, kernel_size, padding='same',
                  activation=act, kernel_initializer='he_normal',
                  name='conv2_2')(inner)
    if norm:
        inner = BatchNormalization()(inner)
    inner = Conv2D(conv_filters2, kernel_size, padding='same',
                activation=act, kernel_initializer='he_normal',
                name='conv2_3')(inner)
    if norm:
        inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters2)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    if (rnn=='cudnn'):
        gru_bidir_1= Bidirectional(CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru_bidir_1'), merge_mode='sum')(inner)
        inner = SpatialDropout1D(p_dropout2)(gru_bidir_1)
        gru_bidir_2= Bidirectional(CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru_bidir_2'), merge_mode='concat')(inner)
    elif (rnn=='gru'):
        gru_bidir_1= Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru_bidir_1', dropout=p_dropout1), merge_mode='sum')(inner)
        gru_bidir_2= Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru_bidir_2', dropout=p_dropout2), merge_mode='concat')(gru_bidir_1)
    else:
        print('Pick valid RNN architecture - gru or cudnn')
        return
    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(gru_bidir_2)
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred)#.summary()

    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer(p_optimizer))
    with open(working_dir + '/config.txt', 'w') as file:
        file.write('batch_size %d \n' % batch_size)
        file.write('RNN_size %d \n' % rnn_size)
        file.write('conv_filters %d \n' % conv_filters)
        file.write('kernel_size  (%d, %d) \n' % kernel_size)
        file.write('pool_size %d \n' % pool_size)
        file.write('time_dense_size %d \n' % time_dense_size)
        file.write('syntetic samples %d \n' % tiger_train.synt_amount)
        file.write('optimizer %s \n' % p_optimizer)
        file.write('dropout in gru1 %f \n' % p_dropout1)
        file.write('dropout in gru2 %f \n' % p_dropout2)
        # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    model.fit_generator(generator=tiger_train.next_batch(),
                            steps_per_epoch=1000,
                            epochs=500,
                            validation_data=tiger_val.next_batch(),
                            validation_steps=tiger_val.n,
                            callbacks=[checkpointer, tb, earlystop, reduce_lr])

    keras.backend.clear_session()
    #return model

#model = train(64, args['dropout'], args['dirpath'], args['optimizer'], 256, 0)

#model = train(128, 0.0, 0.2, 'test', 'adam', 512, 12000, 'cudnn', True)
#model = train(128, 0.0, 0.2, '31.07_8conv_adam_12k_cudnn_dropout_0.0_0.2_LR_plateu_batchnorm1', 'adam', 512, 12000, 'cudnn', True)
#model = train(128, 0.0, 0.2, '31.07_8conv_adam_12k_gru_dropout_0.0_0.2_LR_plateu_batchnorm1', 'adam', 512, 12000, 'gru', True)
model = train(128, 0.0, 0.2, '01.08_8conv_adam_50k_cudnn_dropout_0.0_0.2_LR_plateu_batchnorm', 'adam', 512, 50000, 'cudnn', True)
model = train(128, 0.0, 0.2, '01.08_8conv_adam_50k_cudnn_dropout_0.0_0.2_LR_plateu', 'adam', 512, 50000, 'cudnn', False)
model = train(128, 0.0, 0.2, '01.08_8conv_adam_100k_cudnn_dropout_0.0_0.2_LR_plateu_batchnorm', 'adam', 512, 100000, 'cudnn', True)
model = train(128, 0.0, 0.2, '01.08_8conv_adam_100k_cudnn_dropout_0.0_0.2_LR_plateu', 'adam', 512, 100000, 'cudnn', False)
#model = train(128, 0.0, 0.2, '31.07_8conv_adam_12k_cudnn_dropout_0.0_0.2_LR_plateu2', 'adam', 512, 12000, 'cudnn', False)
#model = train(128, 0.0, 0.2, '31.07_8conv_adam_12k_gru_dropout_0.0_0.2_LR_plateu1', 'adam', 512, 12000, 'gru', False)

#model = train(128, 0.0, 0.2, '31.07_8conv_adam_8k_gru_dropout_0.0_0.2_LR_plateu_batchnorm1', 'adam', 512, 8000, 'gru', True)

#model = train(128, 0.0, 0.2, '31.07_8conv_adam_8k_gru_dropout_0.0_0.2_LR_plateu1', 'adam', 512, 8000, 'gru', False)

#model = train(128, 0.0, 0.2, '31.07_8conv_adam_4k_gru_dropout_0.0_0.2_LR_plateu_batchnorm1', 'adam', 512, 4000, 'gru', True)

#model = train(128, 0.0, 0.2, '31.07_8conv_adam_4k_gru_dropout_0.0_0.2_LR_plateu1', 'adam', 512, 4000, 'gru', False)
