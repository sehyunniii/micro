import cv2
import numpy as np
import tensorflow.lite as tflite
import glob

# ==============================
# 🎯 TensorFlow Lite 모델 로드
# ==============================
interpreter = tflite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 감정 레이블 (모델 학습 순서에 맞게 조정)
emotion_labels = ['행복', '슬픔', '화남', '놀람']


def find_camera_device():
    """USB 또는 CSI 카메라 자동 탐색"""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    print("❌ 카메라 장치를 찾을 수 없습니다.")
    return None


def predict_emotion(face_img):
    """입력된 얼굴 이미지(48x48)로 감정 예측"""
    img = face_img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return emotion_labels[int(np.argmax(preds))]


def get_emotion_from_face():
    """카메라에서 얼굴을 인식하고 감정을 추출"""
    cam_index = find_camera_device()
    if cam_index is None:
        return None

    cap = cv2.VideoCapture(cam_index)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("🎥 라즈베리파이 카메라 감정 인식 중... (ESC 눌러 종료)")

    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 카메라 프레임을 읽을 수 없습니다.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            emotion = predict_emotion(face)
            detected_emotion = emotion

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Emotion Detection (Raspberry Pi)", frame)

        # ESC 키(27)로 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if detected_emotion:
        print(f"✅ 감정 인식 완료: {detected_emotion}")
        return detected_emotion
    else:
        print("😢 감정을 인식하지 못했습니다.")
        return None
