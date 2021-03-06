import cv2
import tensorflow.keras
import numpy as np
import serial
import keyboard

ser = serial.Serial(port = 'COM10', baudrate = 9600)

## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

## 학습된 모델 불러오기
model_filename = 'teachable machine_output\keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(1)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
mode = 'q'
a = 1
b = 0

while True:
    ret, frame = capture.read()
    if ret == True:
        ret, frame = capture.read()
        # 이미지 출력
        cv2.imshow("VideoFrame", frame)

        # 1초마다 검사하며, videoframe 창으로 아무 키나 누르게 되면 종료
        if cv2.waitKey(1):  #원래 200
            break

        # 데이터 전처리
        preprocessed = preprocessing(frame)

        # 예측
        prediction = model.predict(preprocessed)

        # 예측 결과 중 가장 높은 결과 선택
        max_index = np.argmax(prediction[0])
        result = 2
        arr = [1, 2, 3]
        result = arr[max_index]

        # 출력
        print(prediction[0, max_index]*100, '%')
        result = result.encode()
        ser.write(result)
        print(result)
    else:
        print("Wait...")

    
# 카메라 객체 반환
capture.release() 
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()