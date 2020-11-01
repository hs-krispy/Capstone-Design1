import cv2
import find_face

capture = cv2.VideoCapture(0)

while True:
    find = False
    while not find:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
        find = find_face.img_processing('./sv_img/face.jpg', frame)

        if cv2.waitKey(1) > 0:
            break

    print("find face please enter next person")
    sig = input()
    if sig == 'exit':
        break


capture.release()
cv2.destroyAllWindows()