from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np

class CNN_model:
    def __init__(self, x_train, y_train,x_val, y_val, test_data):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.test_data = test_data

    def train(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=(5, 5),strides=(1, 1), padding='same', activation='relu', input_shape=(128, 128, 3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        self.hist = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val,self.y_val), epochs=1, verbose=1)

    def plot_hist(self):
        plt.plot(self.hist.history['loss'])
        plt.show()
        plt.plot(self.hist.history['accuracy'])
        plt.show()

    def predict(self):
        self.pred = self.model.predict(self.test_data)
        return self.pred
