import cv2, numpy as np
from tensorflow.keras.models import load_model

# 모델 라벨 순서 (현재 가정)
emotion_labels = ['슬픔', '행복', '화남', '놀람']
model = load_model("best_model.h5")

# 테스트할 이미지 경로
img = cv2.imread("static/uploads/39843011-angry-face-man.jpg")
if img is None:
    print("⚠️ 이미지 파일을 찾을 수 없습니다.")
else:
    # 전처리
    face = cv2.resize(img, (96, 96))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    # 예측
    preds = model.predict(face)[0]

    print("\n[DEBUG] 감정 확률:")
    for i, p in enumerate(preds):
        print(f"{emotion_labels[i]}: {p:.4f}")

    print(f"[DEBUG] 최종 감정: {emotion_labels[np.argmax(preds)]}\n")

    # 각 인덱스별 매핑 확인
    print("[DEBUG] 예측 순서별 확률 매핑:")
    for i, label in enumerate(['슬픔', '행복', '화남', '놀람']):
        print(f"{i} → {label}: {preds[i]:.4f}")
