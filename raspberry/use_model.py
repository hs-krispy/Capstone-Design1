from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def standard_scaler(x_data):
    mean = np.mean(x_data)
    std = np.std(x_data)
    scale_data = (x_data - mean) / std
    return scale_data

def predict_mask(model):
    img = image.load_img('./sv_img/face.jpg', target_size=(128, 128))
    img_tensor = image.img_to_array(img)
    scale_img = standard_scaler(img_tensor)
    scale_img = scale_img.reshape(-1, 128, 128, 3)
    pred = model.predict_classes(scale_img)
    return pred
