import cv2
import find_face
import use_model
from keras.models import load_model

white_model = load_model('white_mask.h5')
capture = cv2.VideoCapture(0)

while True:
    find = False
    while not find:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
        find = find_face.img_processing('./sv_img/face.jpg', frame)

        if cv2.waitKey(1) > 0:
            break
    pred = use_model.predict_mask(white_model)
    print(pred)
    if pred == 1:
        print("pass")
    elif pred == 0:
        print('please wear a mask')

    else:
        print('aa')

    print("find face please enter next person")

    sig = input()
    if sig == 'exit':
        break


capture.release()
cv2.destroyAllWindows()