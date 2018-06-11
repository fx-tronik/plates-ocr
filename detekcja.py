
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
from os.path import join
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)


letters = sorted([' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3',
           '4', '5', '6', '7', '8', '9', '0'])


class predict:

    incorrect = 0

    def __init__(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        self.model = load_model('weights_08-0.12.hdf5', custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})

    def collect_data(self, dirpath):
        files = os.listdir(dirpath)
        self.img_pre = np.ones([len(files), 60, 260], dtype=np.uint8)
        self.X_data = np.ones([len(files), 260, 60, 1])
        self.filenames = []
        for index, file in enumerate(files):
            filename = file.split('.', 1)[0]
            self.filenames.append(filename)
            img_path = join(dirpath, file)
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (260, 60))
            self.img_pre[index] = img
            img = (img.astype(np.float32) / 255)
            img = np.expand_dims(img.T, axis=2)
            self.X_data[index] = img

    def single_picture(self, picpath):
        self.X_data = np.ones([1, 260, 60, 1])
        self.img_pre = cv2.imread(picpath, 0)
        img = cv2.resize(self.img_pre, (260, 60))
        img = (img.astype(np.float32) / 255)
        img = np.expand_dims(img.T, axis=2)
        self.X_data[0] = img

    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(letters):
                    outstr += letters[c]
            ret.append(outstr)
        return ret

    def decode(self):

        net_inp = self.model.get_layer(name='the_input').input
        net_out = self.model.get_layer(name='softmax').output
        net_out_value = self.sess.run(net_out, feed_dict={net_inp: self.X_data})
        self.pred_texts = self.decode_batch(net_out_value)

    def calculate_distance(self, index):
        self.pred_texts[index] = self.pred_texts[index].replace(" ", "")
        self.filenames[index] = self.filenames[index].replace(" ", "")
        print("----------------------------------")
        print(self.pred_texts[index] + " :przewidywany wynik")
        print(self.filenames[index] + " :prawidlowy wynik")

        distance = Levenshtein.distance(self.filenames[index], self.pred_texts[index])

        if (distance != 0):
            self.incorrect += 1
            print("Blad!")

        print("Dystans: " + str(distance))
        print("Dotychczas zle: " + str(self.incorrect))

    def calculate_accuracy(self):

        for index, prediction in enumerate(self.pred_texts):
            self.calculate_distance(index)

        self.samples = len(self.filenames)
        print("----------------------------------")
        print("Razem probek: " + str(self.samples))
        print("Blednych probek: " + str(self.incorrect))
        print("f score :" + str((self.samples-self.incorrect)/self.samples))

    def display_results(self):

        for index, prediction in enumerate(self.pred_texts):
            self.calculate_distance(index)
            cv2.imshow('image', self.img_pre[index])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def single_result(self):
        try:
            print(self.pred_texts)
            cv2.imshow('image', self.img_pre)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except AttributeError:
            print('Najpierw przeprowadz detekcje')

    def return_raw(self):
        try:
            return self.pred_texts
        except AttributeError:
            print('Najpierw przeprowadz detekcje')


test = predict()

dirpath = '/home/yason/ocr/img/val'
# dirpath = '/home/yason/tablice_kilka_nowych'
# picpath = '/home/yason/tablice_kilka_nowych/DW 7N777.jpg'
# test.single_picture(picpath)

test.collect_data(dirpath)
test.decode()
# test.single_result()
test.calculate_accuracy()
# test.display_debug()
