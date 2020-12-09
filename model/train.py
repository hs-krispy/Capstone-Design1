from model import CNN_model
import numpy as np


x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

model = CNN_model(x_train, y_train, x_val, y_val, x_test)
model.train()
pred = model.predict()
for i in range(6):
    print(pred[i], y_test[i])
