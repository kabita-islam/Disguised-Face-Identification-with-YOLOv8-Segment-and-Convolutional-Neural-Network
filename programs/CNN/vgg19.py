# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 00:15:24 2023

@author: Plabon Dibra
"""

########################### SELECTED ##################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

#setting up GPU
'''
gpus = tf.config.experimental.list_physical_devices('GPU');
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu,True)
'''

train_path = 'F:/Thesis_CE18030&60/resource/CNN/Dataset/train'
valid_path = 'F:/Thesis_CE18030&60/resource/CNN/Dataset/validation'
#test_path = 'F:/Thesis_CE18030&60/resource/CNN/Dataset/test'

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = train_path,
    labels='inferred',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (256,256)
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory = valid_path,
    labels='inferred',
    label_mode = 'categorical',
    batch_size = 32,
    image_size = (256,256)
)

#Normalization
def process(image,label):
    image = tf.cast(image/255. , tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)


#VGG16 model
IMAGE_SIZE =[256,256]
vgg19 = VGG19(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
for layer in vgg19.layers:
    layer.trainable = False

from tensorflow.keras.layers import Dropout
x = Flatten()(vgg19.output)
#x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout with a rate of 0.5

x = Dense(4096, activation='relu')(x)
#x = Dropout(0.5)(x)  # Adding dropout with a rate of 0.5

prediction = Dense(9, activation='softmax')(x)

model = Model(inputs = vgg19.input, outputs=prediction)
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_ds,epochs=200,validation_data=validation_ds)

 

plt.title("accurcy")
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()



plt.title("loss")
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()
 
model.save("F:/Thesis_CE18030&60/resource/CNN/vgg19.h5")




































