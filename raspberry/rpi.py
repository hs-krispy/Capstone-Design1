import cv2
import find_face
import use_model
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
import time
import serial

ser = serial.Serial('/dev/ttyACM0', 9600)
model = load_model('final_mask.h5')
capture = VideoStream(usePiCamera=True, resolution=(640, 480)).start()
time.sleep(2.0)

while True:
    find = False
    frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    find = find_face.img_processing('./sv_img/face.jpg', frame)
    if cv2.waitKey(1) > 0:
        break

    if not find:
        continue

    pred = use_model.predict_mask(model)
    if pred == 1:
        print("pass")
        c = 'y'
        c = c.encode('utf-8')
        ser.write(c)
        while True:
            if ser.readable():
                res = ser.readline()
                print(res.decode()[:len(res) - 1])
            break

    else:
        print('please wear a mask')

    start_time = time.time()

    while time.time() - start_time < 5:
        frame = capture.read()
        if pred == 1:
            cv2.putText(frame, "PASS", (200, 480), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 10)
        else:
            cv2.putText(frame, "Please wear a mask", (100, 480), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 10)
        cv2.putText(frame, res.decode()[:len(res) - 1], (0, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 10)
        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0:
            break

capture.release()

cv2.destroyAllWindows()