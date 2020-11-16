import cv2
import find_face
import use_model
from tensorflow.keras.models import load_model
from pyimagesearch.notifications.twilionotifier import TwilioNotifier
from pyimagesearch.utils.conf import Conf
import time

conf = Conf('config/config.json')
tn = TwilioNotifier(conf)

model = load_model('mask_model.h5')
capture = cv2.VideoCapture(0)

while True:
    find = False
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    find = find_face.img_processing('./sv_img/face.jpg', frame)
    if cv2.waitKey(1) > 0:
            break

    if not find:
        continue

    pred = use_model.predict_mask(model)

    if pred == 0:
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                print("[INFO] sending txt message")
                tn.send("someone don't wear a mask")
                print("[INFO] txt message sent")
                break
        # while time.time() - start_time < 5:
        #     ret, frame = capture.read()
        #     cv2.imshow("VideoFrame", frame)
        #     if cv2.waitKey(1) > 0:
        #         break
            # 여기에 테스트 배드 코드


capture.release()
cv2.destroyAllWindows()




