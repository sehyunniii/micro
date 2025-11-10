import tensorflow as tf
import os

# ============================
# 💾 변환할 모델 파일 이름 설정
# ============================
SOURCE_MODEL = "best_model.h5"
TARGET_MODEL = "best_model.tflite"

# ============================
# 🧠 Keras 모델 로드
# ============================
if not os.path.exists(SOURCE_MODEL):
    raise FileNotFoundError(f"❌ {SOURCE_MODEL} 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인하세요.")

print(f"📂 '{SOURCE_MODEL}' 모델 로드 중...")
model = tf.keras.models.load_model(SOURCE_MODEL)

# ============================
# ⚙️ TensorFlow Lite 변환기 생성
# ============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔧 선택적으로 최적화 (속도/용량 개선)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ============================
# 💾 변환 및 저장
# ============================
print("🔄 변환 중... 잠시만 기다려주세요.")
tflite_model = converter.convert()

with open(TARGET_MODEL, "wb") as f:
    f.write(tflite_model)

print(f"✅ 변환 완료! '{TARGET_MODEL}' 파일이 생성되었습니다.")
print("이제 라즈베리파이에서 face_emotion_detector_pi.py와 함께 사용하세요.")
