import cv2
import find_face
import use_model
from keras.models import load_model
import time

model = load_model('mask_model.h5')
#black_model = load_model('black_mask.h5')
capture = cv2.VideoCapture(0)

while True:
    find = False
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    find = find_face.img_processing('./sv_img/face.jpg', frame)
    if cv2.waitKey(1) > 0:
            break

    if not find :
        continue


    pred = use_model.predict_mask(model)

    if pred == 1:
        print("pass")

    else :
        print('please wear a mask')

    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0:
            break
        # 여기에 테스트 배드 코드


capture.release()
cv2.destroyAllWindows()