# SqueezeNet Model Source Code
# Dev. Dongwon Paek

# Image Preprocessing

import h5py
import os, shutil
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras import layers, models, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import SqueezeNet_11

# Diretory Path Creation
original_dataset_dir = '/home/bearpaek/data/datasets/catsAndDogs/train'
base_dir = '/home/bearpaek/data/datasets/catsAndDogsSmall'

try:
    os.mkdir(base_dir)
except FileExistsError:
    print("괜찮아요우")

total_dir = list()

train_dir = os.path.join(base_dir, 'train')
total_dir.append(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
total_dir.append(validation_dir)
test_dir = os.path.join(base_dir, 'test')
total_dir.append(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
total_dir.append(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
total_dir.append(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
total_dir.append(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
total_dir.append(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
total_dir.append(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
total_dir.append(test_dogs_dir)

try:
    for item in total_dir:
        os.mkdir(item)
except:
    print("괜찮아요우")

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print("훈련용 고양이 이미지 전체 개수: ", len(os.listdir(train_cats_dir)))
print("훈련용 강아지 이미지 전체 개수: ", len(os.listdir(train_dogs_dir)))
print("검증용 고양이 이미지 전체 개수: ", len(os.listdir(validation_cats_dir)))
print("검증용 강아지 이미지 전체 개수: ", len(os.listdir(validation_dogs_dir)))
print("테스트용 고양이 이미지 전체 개수: ", len(os.listdir(test_cats_dir)))
print("테스트용 강아지 이미지 전체 개수: ", len(os.listdir(test_dogs_dir)))

sn = SqueezeNet_11(input_shape = (224, 224, 3), nb_classes=2)
sn.summary()
train_data_dir = '/home/bearpaek/data/datasets/catsAndDogsSmall/train'
validation_data_dir = '/home/bearpaek/data/datasets/catsAndDogsSmall/validation'
nb_train_samples = 2000
nb_validation_samples = 1000
nb_epoch = 500
nb_class = 2
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

# Instantiate AccLossPlotter to visualise training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
checkpoint = ModelCheckpoint(                                         
                'weights.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss',                               
                verbose=0,                                        
                save_best_only=True,                              
                save_weights_only=True,                           
                mode='min',                                       
                period=1)                                         

sn.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, 
        callbacks=[checkpoint])

sn.save_weights('weights.h5')

sn.save('catsAndDogs.h5')