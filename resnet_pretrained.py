# -*- coding: UTF-8 -*-
'''resnet_50 pretrained'''
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import load_data as ld


def run_resnet50(batch_size=16,epochs=20,lr=0.0001):
    resnet_weights_path = '/home/songyu/AICA/codes/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


    image_size = 224
    num_classes = 5

    F = ld.Flower()
    flowers,Image,Label,Label_onehot = F.read_img(image_size)
    X_train,y_train,X_val,y_val,X_test,y_test = F.split_data(flowers,Image,Label,Label_onehot,returnwhat=2)
    X_train = np.vstack([X_train,X_val])
    y_train = np.vstack([y_train,y_val])


    model = Sequential()

    model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.layers[0].trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('test loss and accuracy is',score)













