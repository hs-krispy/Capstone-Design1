## Face detection

- 기존의 얼굴인식에 자주 사용되는 haarcascade_frontalface_default.xml를 이용하려고 했으나 이 경우에 마스크를 착용한 사람은 눈과 코가 동시에 인식되지 않아서 얼굴로 인식을 못하는 문제가 발생

- 이에 따라 Opencv의 dnn 모듈을 이용

  딥러닝을 통해 학습된 binary 상태의 모델 파일 res10_300x300_ssd_iter_140000.caffemodel과

  해당 신경망 모델의 레이어를 구성하고 속성을 정의한 deploy.prototxt.txt를 사용해 문제해결

