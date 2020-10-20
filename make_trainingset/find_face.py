import numpy as np
import cv2
import os

def img_processing(path):
    protoPath = "deploy.prototxt"
    modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    mask_list = os.listdir(path)
    for img_name in mask_list:
        if img_name == '.DS_Store':
            continue
        img_path = path + '/' + img_name
        save_path = path + '_p/' + img_name
        image = cv2.imread(img_path)
        h = 360
        w = 480
        image = cv2.resize(image,(w, h))
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                cv2.imwrite(save_path, face)



img_processing('./mask')
img_processing('./nomask')