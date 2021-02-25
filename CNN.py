import glob
import cv2

import numpy as np 
import pandas as pd
from scipy.spatial.distance import pdist, squareform #scipy spatial distance
import sklearn as sk
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU
from keras import metrics
from keras import backend as K
import time
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
import imageio

x_train = np.zeros((len(dataset),win_lenth*2,win_lenth*2))
y_train = np.zeros(len(dataset))
d = 0
c = 0
start = time.time()
class_translate =[]
class_names =[]
for index in range(np.amin(dataset),np.amax(dataset),win_lenth):
    yclass = "c" + str(index)
    filenames = glob.glob("dataCnn/"+ str(yclass)+ "/*.jpg")
    filenames.sort()
    if len(filenames) != 0:
        filen = filenames[0].split('/')[-1].split('.')[0]
        label_class =filen.split('\\')[0]
        for num in range(0,len(filenames),1):
            class_names.append(label_class)
            class_translate.append(d)
        d = d + 1
        # print(class_translate)
        # print(class_names)
    for image_path in filenames:
        img = imageio.imread(image_path)
        # print(img)
        # print(img.dtype)
    # images = [cv2.imread(img) for img in filenames]
    # for img in images:
    #     img.show()
        dat = img
        # dat = resize(img, (20,20),mode='gray')
        x_train[c,:,:] = dat
        # print(c)
        # y_train[c] = class_translate[e-1]
        c = c + 1
    end = time.time()
    print('Elapsed time:')
    print(end - start)
    print('c:' + str(c))
    print('length:' + str(len(x_train)))
    # print(x_train)
    y_train = class_translate
    # print(y_train)
    filenames = []

model = Sequential()

model.add(Convolution2D(win_lenth*2, (3, 3), activation='relu', input_shape=(1, win_lenth*2, win_lenth*2), data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.30))
# model.add(Convolution2D(20, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(64, activation='relu'))
model.add(LeakyReLU(alpha=0.03))
# model.add(Dropout(0.5))
model.add(Dense(62, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#reshape to include depth
X_train = x_train.reshape(x_train.shape[0], 1, win_lenth*2,win_lenth*2)
#convert to float32 and normalize to [0,1]
X_train = X_train.astype('float32')
X_train /= np.amax(X_train)
# convert labels to class matrix, one-hot-encoding
Y_train = np_utils.to_categorical(y_train)
# split in train and test set
# حلقه for 1000 تایی روی کل دیتا
X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

model.fit(X_train, Y_train, epochs=20, batch_size=40,shuffle=True)

predictions= model.predict(x_test)
rounded = [np.argmax(x) for x in predictions]
print(K.eval(metrics.categorical_accuracy(y_test, np_utils.to_categorical(rounded))))