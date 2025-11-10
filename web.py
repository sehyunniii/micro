import speech_recognition as sr
import pyttsx3
import random
import time
import os
import queue
import sounddevice as sd
import vosk
import json

# -------------------------
# 0. 모델 및 카메라/스레딩 라이브러리
# -------------------------
import cv2  # OpenCV (카메라)
import numpy as np  # 숫자 처리
import tensorflow as tf  # Keras 모델
from tensorflow.keras.models import load_model
import threading  # 동시 처리를 위한 스레딩

# -------------------------
# 0-1. Keras 이미지 모델 로드 및 설정
# -------------------------
MODEL_PATH = 'best_model.h5'
# OpenCV 얼굴 탐지기 XML 파일 (코드와 같은 폴더에 있어야 함)
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# -----------------------------------------------------------
# ★★★ (가정 1) 모델이 학습된 입력 크기로 수정하세요. (예: (48, 48) 또는 (224, 224))
MODEL_INPUT_SIZE = (48, 48)
# ★★★ (가정 2) 모델의 감정 레이블 순서를 훈련시킨 순서와 '정확히' 일치하게 수정하세요.
EMOTION_LABELS = ['angry', 'happy', 'sad', 'surprised', 'neutral']
# ★★★ (가정 3) 모델이 흑백(1)로 학습했는지 컬러(3)로 학습했는지 확인하세요.
MODEL_INPUT_CHANNELS = 1  # 흑백이면 1, 컬러면 3
# -----------------------------------------------------------

try:
    # 이미지 감정 분석 모델 로드
    model = load_model(MODEL_PATH)
    print(f"Keras 모델 '{MODEL_PATH}' 로드 성공.")

    # OpenCV 얼굴 탐지기 로드
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"'{FACE_CASCADE_PATH}' 파일을 로드할 수 없습니다.")
    print(f"OpenCV 얼굴 탐지기 '{FACE_CASCADE_PATH}' 로드 성공.")

except Exception as e:
    print(f"오류: 모델 또는 얼굴 탐지기 로드 실패. ({e})")
    print(f"'{MODEL_PATH}'와 '{FACE_CASCADE_PATH}' 파일이 있는지 확인하세요.")
    print("스크립트를 종료합니다.")
    exit(1)

# -------------------------
# 0-2. 실시간 감정 저장을 위한 공용 변수
# -------------------------
current_emotion = "neutral"  # 기본값
emotion_lock = threading.Lock()  # 스레드간 충돌 방지

# -------------------------
# 1. 감정 분석 모듈 (웹캠 스레드)
# -------------------------
def detect_emotions_from_webcam():
    global current_emotion

    cap = cv2.VideoCapture(0)  # 0번 카메라 (기본 웹캠)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return

    print("=== 카메라 감정 인식 시작 ===")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 흑백 또는 컬러 프레임 준비
        if MODEL_INPUT_CHANNELS == 1:
            # 흑백(Grayscale)으로 변환
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # 컬러(BGR) 프레임 사용
            processed_frame = frame

        # 2. 얼굴 찾기 (흑백 변환된 이미지를 사용해야 탐지율이 높음)
        gray_for_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_for_detection, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_emotion_in_frame = "neutral"  # 이번 프레임의 기본값

        for (x, y, w, h) in faces:
            # 3. 얼굴 부분만 자르기 (흑백 또는 컬러 프레임에서)
            face_roi = processed_frame[y:y + h, x:x + w]

            # 4. 모델 입력 크기로 리사이즈 (★ 가정 1 사용)
            resized_face = cv2.resize(face_roi, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)

            # 5. 모델 입력 형태로 변환 (정규화 및 차원 확장)
            normalized_face = resized_face / 255.0

            if MODEL_INPUT_CHANNELS == 1:
                # 흑백 모델: (1, 48, 48, 1)
                input_image = np.expand_dims(np.expand_dims(normalized_face, -1), 0)
            else:
                # 컬러 모델: (1, 48, 48, 3)
                input_image = np.expand_dims(normalized_face, 0)

            # 6. 모델 예측
            # (예측 속도를 높이려면 .predict() 대신 model() 사용 가능)
            try:
                prediction = model(input_image, training=False)
                pred_index = np.argmax(prediction[0])

                # 7. 감정 텍스트로 변환 (★ 가정 2 사용)
                detected_emotion_in_frame = EMOTION_LABELS[pred_index]
            except Exception as e:
                print(f"모델 예측 오류: {e}")
                detected_emotion_in_frame = "neutral"


            # (시각화) 카메라 화면에 사각형과 감정 텍스트 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion_in_frame, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 첫 번째로 감지된 얼굴의 감정만 사용
            break

        # 8. (스레드 안전) 공용 변수에 현재 감정 업데이트
        with emotion_lock:
            current_emotion = detected_emotion_in_frame

        # (시각화) 카메라 창 띄우기
        cv2.imshow('Emotion Detection', frame)

        # 'q' 키 누르면 종료 (웹캠 창에서)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=== 카메라 감정 인식 종료 ===")


# -------------------------
# 2️⃣ 시나리오 매핑 (기존과 동일)
# -------------------------
scenarios = {
    "happy": [
        "좋아요! 오늘 기분이 좋다니 다행이네요.",
        "즐거운 일이 있었나 봐요! 계속 행복하세요."
    ],
    "sad": [
        "힘든 일이 있었군요. 괜찮아요, 조금 쉬어도 돼요.",
        "슬픈 기분이 느껴져요. 제가 옆에서 들어줄게요."
    ],
    "angry": [
        "화가 나셨군요. 깊게 숨을 쉬고 차분히 생각해볼까요?",
        "속상하셨겠어요. 잠시 진정한 후 이야기해요."
    ],
    "surprised": [
        "정말 놀라셨군요! 대단한 일이네요.",
        "예상치 못한 일이 있었나 봐요!"
    ],
    "neutral": [
        "그렇군요. 더 이야기해 볼까요?",
        "알겠습니다. 계속 말씀해주세요."
    ]
}

# -------------------------
# 3️⃣ TTS 초기화 (기존과 동일)
# -------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 말 속도
engine.setProperty('volume', 1.0)  # 볼륨 0.0~1.0

# -------------------------
# 4️⃣ STT 초기화 (기존과 동일 - Vosk 루프에서는 사용 안 됨)
# -------------------------
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# -------------------------
# 4️⃣ 상담 종료 키워드 (기존과 동일)
# -------------------------
exit_keywords = ["끝", "종료", "그만", "안녕"]

# -------------------------
# 5️⃣ Vosk STT 초기화 (Keras 모델과 충돌 방지 위해 변수명 변경)
# -------------------------
vosk_model_path = "vosk-model-small-ko-0.22"  # 한국어 모델 경로
if not os.path.exists(vosk_model_path):
    print("Vosk 모델이 없습니다. https://alphacephei.com/vosk/models 에서 다운로드 후 압축 해제하세요.")
    exit(1)

vosk_model = vosk.Model(vosk_model_path)  # Keras의 'model'과 충돌 방지
q = queue.Queue()

# -------------------------
# USB 마이크 장치 확인 후 device_index 설정 (기존과 동일)
# -------------------------
MIC_DEVICE_INDEX = 1  # USB 마이크 장치 index (확인 후 변경)

def audio_callback(indata, frames, time_, status):
    if status:
        print(status)
    q.put(bytes(indata))

# -------------------------
# 6️⃣ 실시간 상담 루프 (메인 스레드)
# -------------------------

# 1. 웹캠 스레드 시작
# (daemon=True: 메인 스레드가 종료되면 웹캠 스레드도 자동 종료)
cam_thread = threading.Thread(target=detect_emotions_from_webcam, daemon=True)
cam_thread.start()

print("카메라 스레드 시작. 잠시 기다려주세요...")
time.sleep(3)  # 카메라 켜지는 시간 대기

engine.say("안녕하세요. 상담을 '표정'과 '음성'으로 시작합니다. 말씀해주세요.")
engine.runAndWait()

try:
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback,
                           device=MIC_DEVICE_INDEX):

        rec = vosk.KaldiRecognizer(vosk_model, 16000)
        print("=== 상담 시스템 시작 (USB 마이크 + 스피커) ===")
        print("=== (웹캠 창에서 'q'를 누르거나, 터미널에서 Ctrl+C로 종료) ===")

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                result_dict = json.loads(result)
                user_text = result_dict.get("text", "").strip()  # 공백 제거

                if not user_text:
                    continue

                print(f"사용자 입력 (음성): '{user_text}'")

                # 상담 종료 키워드 감지
                if any(word in user_text for word in exit_keywords):
                    print("상담을 종료합니다. 안녕히 가세요!")
                    engine.say("상담을 종료합니다. 안녕히 가세요!")
                    engine.runAndWait()
                    break

                # 2. 웹캠 스레드가 분석한 '현재 감정' 가져오기
                with emotion_lock:
                    emotion = current_emotion
                print(f"감정 감지 (표정): '{emotion}'")

                # 시나리오 매핑 (표정 감정 기반)
                reply_text = random.choice(scenarios.get(emotion, scenarios["neutral"]))

                # 텍스트 키워드 기반 세부 시나리오 (텍스트 + 표정)
                if "친구" in user_text and emotion == "angry":
                    reply_text = "친구와 싸우셨군요. 화가 많이 나셨겠어요. 잠시 진정한 후 이야기를 해볼까요?"
                elif "시험" in user_text and emotion == "sad":
                    reply_text = "시험 때문에 속상하셨군요. 노력한 만큼 결과가 안 나와서 힘드시죠."
                elif "여행" in user_text and emotion == "happy":
                    reply_text = "여행 가시는군요! 정말 신나시겠어요!"

                print(f"응답 시나리오: '{reply_text}'")

                # TTS 출력
                engine.say(reply_text)
                engine.runAndWait()

except KeyboardInterrupt:
    print("사용자에 의해 종료되었습니다.")
except Exception as e:
    print("오류 발생:", e)