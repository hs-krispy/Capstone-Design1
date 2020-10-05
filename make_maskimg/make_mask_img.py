import cv2
import face_recognition
import math
import os


def find_location(nose_bridge, chin):
    nose_bridge = sorted(nose_bridge, key = lambda x : x[1])
    min_y = nose_bridge[2][0]

    min_x = math.inf
    max_x = -math.inf
    max_y = -math.inf

    for x, y in chin:
        if min_x > x:
            min_x = x
        if max_x < x:
            max_x = x
        if max_y < y:
            max_y = y

    return min_x, max_x, min_y, max_y
img_list = os.listdir('img_set')
mask = cv2.imread('blue-mask.png')

for img_name in img_list:
    path = './img_set/' + img_name
    save_path = './mask_img/' + img_name
    image = face_recognition.load_image_file(path)
    face = cv2.imread(path)
    face_landmarks = face_recognition.face_landmarks(image)[0]
    min_x, max_x, min_y, max_y = find_location(face_landmarks['nose_bridge'], face_landmarks['chin'])
    width = max_x - min_x
    hight = max_y - min_y

    mask = cv2.resize(mask, dsize=(width, hight), interpolation=cv2.INTER_LINEAR)
    for i in range(width):
        for j in range(hight):
            if sum(mask[j,i]) <= 300:
                continue
            else:
                face[min_y + j, min_x + i] = mask[j, i]
    cv2.imwrite(save_path, face)