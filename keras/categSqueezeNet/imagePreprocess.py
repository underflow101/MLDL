# SqueezeNet V1.1 Model Source Code
# Run this code with $ python imagePreprocess.py on console
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
from model import SqueezeNet

###############################################################################
# Diretory Path Creation
# Can separate this code to different python file and just import to this
original_dataset_dir = '/home/bearpaek/data/datasets/lpl224/'

origin_others_dir = '/home/bearpaek/data/datasets/lpl224/others'
origin_writing_dir = '/home/bearpaek/data/datasets/lpl224/writing'
origin_phoneWithHand_dir = '/home/bearpaek/data/datasets/lpl224/phoneWithHand'
origin_sleep_dir = '/home/bearpaek/data/datasets/lpl224/sleep'

base_dir = '/home/bearpaek/data/datasets/lplSmall'

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

train_others_dir = os.path.join(train_dir, 'others')
total_dir.append(train_others_dir)
train_writing_dir = os.path.join(train_dir, 'writing')
total_dir.append(train_writing_dir)
train_phoneWithHand_dir = os.path.join(train_dir, 'phoneWithHand')
total_dir.append(train_phoneWithHand_dir)
train_sleep_dir = os.path.join(train_dir, "sleep")
total_dir.append(train_sleep_dir)

validation_others_dir = os.path.join(validation_dir, 'others')
total_dir.append(validation_others_dir)
validation_writing_dir = os.path.join(validation_dir, 'writing')
total_dir.append(validation_writing_dir)
validation_phoneWithHand_dir = os.path.join(validation_dir, 'phoneWithHand')
total_dir.append(validation_phoneWithHand_dir)
validation_sleep_dir = os.path.join(validation_dir, 'sleep')
total_dir.append(validation_sleep_dir)

try:
    for item in total_dir:
        os.mkdir(item)
except:
    print("괜찮아요우")

# Others Training & Validation
fnames = ['{}.jpg'.format(i) for i in range(18000)]
for fname in fnames:
    src = os.path.join(origin_others_dir, fname)
    dst = os.path.join(train_others_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(18000, 20000)]
for fname in fnames:
    src = os.path.join(origin_others_dir, fname)
    dst = os.path.join(validation_others_dir, fname)
    shutil.copyfile(src, dst)

# Writing Training & Validation
fnames = ['{}.jpg'.format(i) for i in range(18000)]
for fname in fnames:
    src = os.path.join(origin_writing_dir, fname)
    dst = os.path.join(train_writing_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(18000, 20000)]
for fname in fnames:
    src = os.path.join(origin_writing_dir, fname)
    dst = os.path.join(validation_writing_dir, fname)
    shutil.copyfile(src, dst)

# phoneWithHand Training & Validation
fnames = ['{}.jpg'.format(i) for i in range(18000)]
for fname in fnames:
    src = os.path.join(origin_phoneWithHand_dir, fname)
    dst = os.path.join(train_phoneWithHand_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(18000, 20000)]
for fname in fnames:
    src = os.path.join(origin_phoneWithHand_dir, fname)
    dst = os.path.join(validation_phoneWithHand_dir, fname)
    shutil.copyfile(src, dst)

# Sleep Training & Validation
fnames = ['{}.jpg'.format(i) for i in range(18000)]
for fname in fnames:
    src = os.path.join(origin_sleep_dir, fname)
    dst = os.path.join(train_sleep_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(18000, 20000)]
for fname in fnames:
    src = os.path.join(origin_sleep_dir, fname)
    dst = os.path.join(validation_sleep_dir, fname)
    shutil.copyfile(src, dst)

print("훈련용 others 이미지 전체 개수: ", len(os.listdir(train_others_dir)))
print("훈련용 필기 이미지 전체 개수: ", len(os.listdir(train_writing_dir)))
print("훈련용 핸드폰사용 이미지 전체 개수: ", len(os.listdir(train_phoneWithHand_dir)))
print("훈련용 엎드려수면 이미지 전체 개수: ", len(os.listdir(train_sleep_dir)))

print("검증용 others 이미지 전체 개수: ", len(os.listdir(validation_others_dir)))
print("검증용 필기 이미지 전체 개수: ", len(os.listdir(validation_writing_dir)))
print("검증용 핸드폰사용 이미지 전체 개수: ", len(os.listdir(validation_phoneWithHand_dir)))
print("검증용 엎드려수면 이미지 전체 개수: ", len(os.listdir(validation_sleep_dir)))
########################################################################################

########################################################################################
# Start learning, as well as compiling model
sn = SqueezeNet(input_shape = (224, 224, 3), nb_classes=4)
sn.summary()
train_data_dir = '/home/bearpaek/data/datasets/lplSmall/train'
validation_data_dir = '/home/bearpaek/data/datasets/lplSmall/validation'
nb_train_samples = 72000
nb_validation_samples = 8000
nb_epoch = 2000
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
early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0)
checkpoint = ModelCheckpoint(                                         
                'weights.{epoch:02d}.h5',
                monitor='val_acc',
                verbose=1,                                        
                save_best_only=True,                              
                save_weights_only=True,                           
                mode='max',                                       
                period=1)                                
###########################################################################
sn.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples, 
        callbacks=[checkpoint]
)

print("Training Ended")

sn.save_weights('weights.h5')
print("Saved weight file")

sn.save('lpl.h5')
print("saved model file")

# End of Code
##########################################################################
