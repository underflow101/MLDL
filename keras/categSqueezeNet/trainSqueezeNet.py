# SqueezeNet V1.1 Model Source Code
# Dev. Dongwon Paek

# Train & Save .h5 file

import h5py
import os, shutil
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras import layers, models, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import SqueezeNet

# Start learning, as well as compiling model
sn = SqueezeNet(input_shape = (224, 224, 3), nb_classes=4)
sn.summary()
train_data_dir = '/home/bearpaek/data/datasets/lplSmall/train'
validation_data_dir = '/home/bearpaek/data/datasets/lplSmall/validation'
nb_train_samples = 72000
nb_validation_samples = 8000
nb_epoch = 500
nb_class = 4
width, height = 224, 224

sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
sn.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

#   Generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')
############################################################################
# Inlcude this Callback checkpoint if you want to make .h5 checkpoint files
# May slow your training
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
#checkpoint = ModelCheckpoint(                                         
#                'weights.{epoch:02d}-{val_loss:.2f}.h5',
#                monitor='val_loss',                               
#                verbose=0,                                        
#                save_best_only=True,                              
#                save_weights_only=True,                           
#                mode='min',                                       
#                period=1)                                
###########################################################################
sn.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples#, 
        #callbacks=[checkpoint])
)

print("Training Ended")

sn.save_weights('weights.h5')
print("Saved weight file")

sn.save('lpl.h5')
print("saved model file")

# End of Code
##########################################################################
