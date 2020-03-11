# modelTest.py

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from models import model_from_json
from keras.preprocessing import image
import numpy as np
from numpy import argmax
from model import SqueezeNet

width, height = 224, 224
base_dir = "./"

total_dir = list()

test_dir = os.path.join(base_dir, 'test')
total_dir.append(test_dir)

test_others_dir = os.path.join(train_dir, 'others')
total_dir.append(test_others_dir)
test_writing_dir = os.path.join(train_dir, 'writing')
test_dir.append(test_writing_dir)
train_phoneWithHand_dir = os.path.join(train_dir, 'phoneWithHand')
test_dir.append(test_phoneWithHand_dir)
train_sleep_dir = os.path.join(train_dir, "sleep")
test_dir.append(test_sleep_dir)

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

test_image = image.load_img("./test/", target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict_classes(test_image)
print(np.argmax(result))
print(result)
{'others': 0, 'writing': 1, 'phoneWithHand': 2, 'sleep': 3}
if result[0][0] == 0:
    prediction = 'others'
elif result[0][0] == 1:
    prediction = 'writing'
elif result[0][0] == 2:
    prediction = 'phoneWithHand'
else:
    prediction = 'sleep'
    
print("True: " + prediction)

for _ in range(1000)