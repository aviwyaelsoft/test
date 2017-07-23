# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:16:32 2017

@author: yael
"""

#BUILDING OUR CNN

#import keras packages for cnn
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D , ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout


# Creates a graph.

#initializing our cnn
classifier = Sequential()

#step 1 Convulution 
#we are need to determine - how many "filters" or feature ditector we want for our
#classifier. and this is very important step. that will create our feature maps (as numbered as our features)
#"filters" is the number of filters we want. and the "kernel_size" is the rows and
#collums its has like [3,3] or one int for same rows and col

#input shape is the size of our picture and we need all of our pictures to be 
#the same shape 
#the shape is for colored 3 on the 2d size picture like [3,1000,800]
#and for black and white [1,500,500] 
#in colored every single sell of the 2d picture has 3 colors
#and 1 with black and white
#here we change it to smaller for it be easy.
#the order in tensorflow backend is different its 64 ,64 ,3 
#we use relu function for we want get negative numbers
classifier.add(ZeroPadding2D((1,1),input_shape=(128,128,3)))
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))

classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, 3, 3, activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, 3, 3, activation='relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Convolution2D(256, 3, 3, activation='relu'))
classifier.add(MaxPooling2D((2,2), strides=(2,2)))



#flattening  - putting all the maps in one big array (vector)
classifier.add (Flatten())

#full connection - (making the ANN)
classifier.add (Dense(128 , activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add (Dense(128 , activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add (Dense(1 , activation = 'sigmoid'))

#compiling our cnn
classifier.compile (optimizer = 'rmsprop' , loss= 'binary_crossentropy' , metrics = ['accuracy'])

#fitting our images with keras help 
#we going to make image argumantaiton.. because we dont have a lot of training set
#so we need more images so we gonna produce more images so the computer can find corralation
from keras.preprocessing.image import ImageDataGenerator

# the rescale - is changing the values of the pixels between 0 to one
# the others are some random things to generate dii pictures

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=250,
        epochs=25,
        validation_data=test_set,
        validation_steps=65)
