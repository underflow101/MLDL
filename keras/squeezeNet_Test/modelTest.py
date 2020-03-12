# modelTest.py

import h5py
import os, shutil
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential

from keras.layers import Dense, Activation
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import argmax
from model import SqueezeNet

base_dir = "./"

total_dir = list()

test_dir = os.path.join(base_dir, 'testImage')
total_dir.append(test_dir)

test_others_dir = os.path.join(test_dir, 'others')
total_dir.append(test_others_dir)
test_writing_dir = os.path.join(test_dir, 'writing')
total_dir.append(test_writing_dir)
test_phoneWithHand_dir = os.path.join(test_dir, 'phoneWithHand')
total_dir.append(test_phoneWithHand_dir)
test_sleep_dir = os.path.join(test_dir, "sleep")
total_dir.append(test_sleep_dir)

print("테스트용 others 이미지 전체 개수: ", len(os.listdir(test_others_dir)))
print("테스트용 필기 이미지 전체 개수: ", len(os.listdir(test_writing_dir)))
print("테스트용 핸드폰사용 이미지 전체 개수: ", len(os.listdir(test_phoneWithHand_dir)))
print("테스트용 엎드려수면 이미지 전체 개수: ", len(os.listdir(test_sleep_dir)))

test_data_dir = './test'
nb_test_samples = 2000
nb_class = 4
width, height = 224, 224

model = SqueezeNet(input_shape = (224, 224, 3), nb_classes=4)
model.load_weights('lpl.h5')
model.summary()
print("Weight loaded complete.")

sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled.")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')

res = model.predict_generator(test_generator, steps=5)

print("-- Predict --")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(res)