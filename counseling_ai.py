import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from collections import deque

# --- 1. 설정 초기화 ---

EMOTION_LABELS = ["Angry", "Happy", "Sad", "Surprise"] # 4가지 감정

# TFLite 모델 로드
TFLITE_MODEL_PATH = 'emotion_model.tflite'
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_INPUT_SHAPE = (input_details[0]['shape'][1], input_details[0]['shape'][2])
MODEL_INPUT_CHANNELS = input_details[0]['shape'][3]

# OpenCV 얼굴 탐지기 로드
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

# 카메라 설정
cap = cv2.VideoCapture(0)

# --- 2. 상담 로직 변수 (새로 추가된 부분) ---

# 감정 버퍼 (최근 100 프레임의 감정을 저장)
emotion_buffer = deque(maxlen=100) 

# 상담 상태 변수
current_conversation_state = "Greeting" # 대화의 현재 상태
current_stable_emotion = "None"         # 현재 지속되는 감정
last_interaction_time = time.time()     # 마지막으로 AI가 말한 시간
interaction_cooldown = 10.0             # 10초에 한 번만 말하도록 쿨다운

# --- 3. 전처리 함수 (이전과 동일) ---
def preprocess_face(face_image):
    if MODEL_INPUT_CHANNELS == 1:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, MODEL_INPUT_SHAPE)
    face_image = face_image / 255.0
    if MODEL_INPUT_CHANNELS == 1:
        face_image = np.expand_dims(face_image, axis=-1)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image.astype(np.float32)

# --- 4. 상담 시나리오 함수 (새로 추가) ---
def get_counseling_response(state, emotion):
    """
    현재 대화 상태(state)와 지속적인 감정(emotion)을 기반으로
    AI의 응답(response)과 다음 상태(next_state)를 반환합니다.
    """
    response = ""
    next_state = state # 기본적으로 현재 상태 유지

    # [매우 중요] 이 부분을 원하는 시나리오로 채워넣어야 합니다.
    
    if state == "Greeting":
        response = "안녕하세요. 오늘 기분은 어떠신가요?"
        next_state = "Listening" # 인사를 했으니 '경청' 상태로 변경
        
    elif state == "Listening":
        # 경청 상태에서는 특정 감정이 지속될 때 반응
        if emotion == "Sad":
            response = "조금 슬퍼 보이시네요. 혹시 무슨 일 있으신가요?"
            next_state = "Asking_Sad" # '슬픔 질문' 상태로 변경
        elif emotion == "Happy":
            response = "표정이 정말 밝아 보이세요! 좋은 일이 있으신가요?"
            next_state = "Asking_Happy" # '행복 질문' 상태로 변경
        elif emotion == "Angry":
            response = "무언가 불편한 일이 있으신 것 같아요."
            next_state = "Asking_Angry" # '화남 질문' 상태로 변경

    elif state == "Asking_Sad":
        # '슬픔 질문' 상태에서 사용자가 계속 슬퍼 보인다면...
        if emotion == "Sad":
            response = "편하게 말씀해주셔도 괜찮아요. 저는 듣고 있어요."
            next_state = "Comforting" # '위로' 상태로 변경
        elif emotion == "Neutral" or emotion == "Happy":
            response = "기분이 조금 나아지셨나 봐요. 다른 이야기를 해볼까요?"
            next_state = "Listening" # 다시 '경청' 상태로

    elif state == "Comforting":
        # 위로 상태에서 20초 정도 머무르다가 다시 경청 상태로 돌아감 (타이머 로직 필요)
        # (여기서는 단순화를 위해 일단 대기)
        pass

    # ... (Happy, Angry, Surprise 상태에 대한 시나리오도 추가 ...)
    
    return response, next_state

# --- 5. 메인 루프 (수정됨) ---

# AI가 처음 시작할 때 인사말을 하도록 함
initial_response, current_conversation_state = get_counseling_response("Greeting", "None")
print(f"AI: {initial_response}") # 콘솔에 출력 (나중에 GUI로 변경)
last_interaction_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(30, 30))
    
    instant_emotion = "None" # 현재 프레임의 감정

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            processed_face = preprocess_face(face_roi)
            
            interpreter.set_tensor(input_details[0]['index'], processed_face)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            
            emotion_index = np.argmax(predictions)
            instant_emotion = EMOTION_LABELS[emotion_index] # '순간 감정'
            
            # (시각화) 화면에 '순간 감정' 표시
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, instant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            pass

    # --- 6. 상태 관리 로직 (새로 추가) ---
    
    # 1. 감정 버퍼에 '순간 감정' 추가 (얼굴이 감지되었을 때만)
    if instant_emotion != "None":
        emotion_buffer.append(instant_emotion)

    current_time = time.time()
    
    # 2. 쿨다운이 지났고 (10초), 버퍼가 충분히 찼을 때만 상태 업데이트
    if current_time - last_interaction_time > interaction_cooldown and len(emotion_buffer) == emotion_buffer.maxlen:
        
        # 3. '지속 감정' 계산 (최빈값)
        current_stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        emotion_buffer.clear() # 버퍼 비우기
        
        # 4. 시나리오 엔진 가동
        response, next_state = get_counseling_response(current_conversation_state, current_stable_emotion)
        
        if response: # 응답할 내용이 있다면
            print(f"AI (감정: {current_stable_emotion} | 상태: {current_conversation_state}): {response}")
            current_conversation_state = next_state # 대화 상태 업데이트
            last_interaction_time = current_time # 쿨다운 타이머 리셋

    # (시각화) 현재 '지속 감정'과 '대화 상태'를 화면에 표시
    cv2.putText(frame, f"Stable Emotion: {current_stable_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"AI State: {current_conversation_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow('Counseling AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()