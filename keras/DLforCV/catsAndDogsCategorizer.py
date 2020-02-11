# Cats and Dogs Categorizer

import os, shutil
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

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

# Stack layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
########
#model.add(layers.Conv2D(256, (3, 3), activation = 'relu'))
#model.add(layers.MaxPooling2D(1, 1))
########
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')

for data_batch, labels_batch in train_generator:
    print("배치 데이터 크기: ", data_batch.shape)
    print("배치 레이블 크기: ", labels_batch.shape)
    break

history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30, validation_data = validation_generator, validation_steps = 50)
model.save('catsAndDogsSmall1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

####################################################################

