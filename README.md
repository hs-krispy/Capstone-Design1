## MASK CCTV

##### 대중교통 이용시 얼굴인식을 이용해 마스크를 착용여부를 판별하고 탑승 제한, 경고음, 알림 전송 등 후속 조치를 시행 

- 최근 코로나 확산으로 인하여 마스크 착용이 의무화 되고 있는 중에 마스크를 쓰지 않고 대중교통을 이용하여 문제가 되는 상황이 많이 발생하고 있음
- 승객들이 안심하고 대중교통을 탑승할 수 있도록 하며 이러한 갈등 또한 사전에 방지하기 위해 이와 같은 프로젝트를 진행하고자 함



#### [얼굴 데이터 제공 (generated photos)](https://generated.photos/)

### make_maskimg

- 기존의 얼굴 데이터에 마스크를 합성시켜서 마스크를 착용한 얼굴 데이터를 생성



### make_trainingset

- 데이터를 모델 학습에 적합하게 만들기위해 위에서 얻은 마스크 미착용과 착용 데이터셋에서 얼굴을 인식하고 해당 부분만 추출 후 다시 미착용과 착용 디렉토리로 분류



### model

- 모델 학습을 위해 이미지 파일을 npy 파일로 변환하고 레이블을 지정 **(1 - 마스크 착용, 0 - 마스크 미착용)**
- CNN을 통해 모델을 학습시키고 성능을 테스팅
