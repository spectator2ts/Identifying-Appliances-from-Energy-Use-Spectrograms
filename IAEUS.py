import numpy as np
import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Model, Sequential, load_model
from keras.layers import MaxPooling2D, Dropout, concatenate
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, ReLU, Activation
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, image

trainPath = "/kaggle/input/identifying-appliances-from-energy-use-spectrogram/train"
testPath = "/kaggle/input/identifying-appliances-from-energy-use-spectrogram/test"

# load an example image
expPath = "/kaggle/input/identifying-appliances-from-energy-use-spectrogram/train/1000_c.png"
expImg = image.load_img(expPath, color_mode="grayscale")

# look at the image
plt.imshow(expImg)

# convert img as input format
expImgArray = image.img_to_array(expImg)
print("The shape of the example img input array is {}".format(expImgArray.shape))

def match(trainLabels, trainPath):
"""
Given a list of train labels, return the corresponding train ids.

    @Parms:
    trainLabels: List[int]
    trainPath: String
    
    @Returns:
    x: List[(numInstances, imgLength, imgWidth, numChannels)]
"""
    xCurr = []
    xVol = []
    for i, label in enumerate(trainLabels):
        # join path
        pathCurr = os.path.join(trainPath, '{}_c.png'.format(label))
        pathVol = os.path.join(trainPath, '{}_v.png'.format(label))
        
        # create image input
        imgCurr = image.load_img(pathCurr, color_mode="grayscale")
        imgCurrArray = image.img_to_array(imgCurr)
        
        imgVol = image.load_img(pathVol, color_mode="grayscale")
        imgVolArray = image.img_to_array(imgVol)
        
        # join the instance
        xCurr.append(imgCurrArray)
        xVol.append(imgVolArray)

    x = np.concatenate([np.asarray(xCurr), np.asarray(xVol)], axis=3)

    return x

# load train labels
trainLabels = pd.read_csv("/kaggle/input/identifying-appliances-from-energy-use-spectrogram/train_labels.csv")

# split train and test sets
_len = int(len(trainLabels) * 0.8)
trainLabels, testLabels = trainLabels[: _len], trainLabels[_len: ]
trainListY = list(trainLabels["id"])
testListY = list(testLabels["id"])
trainY = to_categorical(trainLabels["appliance"].values)
testY = to_categorical(testLabels["appliance"].values)

# load train X
trainX = match(trainListY, trainPath)
testX = match(testListY, trainPath)

# modeling
inputs = Input(shape=trainX.shape[1:])

x = Conv2D(64, (2, 3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.01))(inputs)
x = MaxPooling2D(pool_size=(2,3), padding='same')(x)

x = Conv2D(128, (2, 3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.03))(x)
x = MaxPooling2D(pool_size=(2,3), padding='same')(x)

x = Conv2D(128, (2, 3), activation='relu', padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
x = Conv2D(256, (2, 3), activation='relu',padding="same", kernel_regularizer=regularizers.l2(0.03))(x)
x = MaxPooling2D(pool_size=(2,3), padding='same')(x)
x = Dropout(0.5)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization(momentum=0.8)(x)
x = Dropout(0.5)(x)

predictions = Dense(11, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)

ada = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss="categorical_crossentropy", optimizer=ada, metrics=["accuracy"])

# data augmentation
datagen = ImageDataGenerator(width_shift_range=0.4)
datagen.fit(trainX)

# fit model
model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),epochs=5, validation_data=(testX, testY))

# make prediction
submission = pd.read_csv("/kaggle/input/identifying-appliances-from-energy-use-spectrogram/submission_format.csv")
outputID = submission["id"]
outputX = match(outputID, testPath)

outputY = model.predict(outputX)
label_test = np.argmax(outputY, axis=1)
submission["appliance"] = label_test
submission.to_csv('submission_concat.csv', index=False)
