import tensorflow as tf

# 1. 훈련된 Keras 모델 로드 (파일 경로 확인)
try:
    # .h5 파일인 경우
    model = tf.keras.models.load_model('best_model.h5')
except Exception as e:
    # SavedModel 폴더인 경우
    model = tf.keras.models.load_model('your_saved_model_directory')

# 2. TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. (권장) 모델 최적화 (속도 향상)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. 모델 변환
tflite_model = converter.convert()

# 5. .tflite 파일로 저장
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite 모델 변환 완료: 'emotion_model.tflite' 파일이 생성되었습니다.")
print("이 파일을 라즈베리파이로 복사하세요.")