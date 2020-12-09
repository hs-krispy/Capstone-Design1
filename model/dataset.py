from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import os

def make_dataset(path,x_data, y_data):
    img_list = os.listdir(path)
    print(img_list)
    for img_name in img_list:
        if img_name == '.DS_Store':
            continue
        img_path = path + '/' + img_name
        img = image.load_img(img_path, target_size=(128, 128))
        img_tensor = image.img_to_array(img)
        x_data.append(img_tensor)
        if path == './mask_p' or path == './test_mask':
            y_data.append([1])
        else:
            y_data.append([0])

def standard_scaler(x_data):
    mean = np.mean(x_data, axis = 0)
    std = np.std(x_data, axis = 0)
    scale_data = (x_data - mean) / std
    return scale_data

def save_data(x_train, x_val, y_train, y_val):
    np.save('x_train.npy', x_train)
    np.save('x_val.npy', x_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)

x_data = []
y_data = []

make_dataset('./mask_p', x_data, y_data)
make_dataset('./nomask_p', x_data, y_data)
# make_dataset('./test_mask', x_data, y_data)
# make_dataset('./test_nomask', x_data, y_data)

x_data = np.array(x_data, dtype=np.uint8)
y_data = np.array(y_data, dtype=np.uint8)
scale_data = standard_scaler(x_data)

x_train, x_val, y_train, y_val = train_test_split(scale_data, y_data, test_size=0.1, shuffle = True, random_state=1234)
save_data(x_train, x_val, y_train, y_val)