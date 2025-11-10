from flask import Flask, render_template, request, redirect, url_for, session
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import random
import json

# ---------------------------
# Flask 기본 설정
# ---------------------------
app = Flask(__name__)
app.secret_key = "my_secret_key"  # 세션 저장용 키
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ---------------------------
# 모델 및 데이터 로드
# ---------------------------
model = load_model("best_model.h5")
emotion_labels = ['슬픔', '행복', '화남', '놀람']

with open("counseling_data.json", "r", encoding="utf-8") as f:
    counseling_data = json.load(f)


# ---------------------------
# 감정 분석 함수 (얼굴 검출 비활성화)
# ---------------------------
def analyze_emotion(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] 이미지를 불러오지 못했습니다.")
        return None

    face = cv2.resize(img, (96, 96))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    preds = model.predict(face)
    predicted_idx = np.argmax(preds)
    emotion = emotion_labels[predicted_idx]

    print(f"[DEBUG] 예측 감정: {emotion}")
    for i, label in enumerate(emotion_labels):
        print(f"    {label}: {preds[0][i]:.4f}")
    print(f"[DEBUG] 최종 감정: {emotion}\n")

    return emotion


# ---------------------------
# 상담 키워드 탐지
# ---------------------------
def detect_keyword(user_input):
    keywords = {
        "사람": ["친구", "가족", "사람", "연인", "동료"],
        "일": ["회사", "직장", "일", "공부", "시험", "업무"],
        "성취": ["성공", "목표", "달성", "성과"],
        "스트레스": ["스트레스", "짜증", "피곤", "지침"],
        "외로움": ["혼자", "외롭", "고독"],
        "사건": ["사고", "변화", "갑자기"],
        "좋은소식": ["좋은", "기쁜", "행운", "축하"],
        "나쁜소식": ["나쁜", "슬픈", "충격", "불행"],
        "일상": ["하루", "산책", "날씨", "커피"]
    }
    for key, words in keywords.items():
        if any(word in user_input for word in words):
            return key
    return "기타"


# ---------------------------
# 홈 (사진 업로드)
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    session.clear()  # 이전 대화 세션 초기화

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="이미지를 선택해주세요.")

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # 감정 분석
        emotion = analyze_emotion(file_path)
        print(f"[DEBUG] Flask로 전달된 감정 결과: {emotion}")

        if emotion:
            session["emotion"] = emotion
            session["chat_history"] = [
                {"sender": "bot", "text": random.choice(counseling_data[emotion]["intro"])}
            ]
            return redirect(url_for("chat", emotion=emotion))
        else:
            return render_template("index.html", error="얼굴을 인식하지 못했습니다. 다른 사진으로 시도해주세요.")
    return render_template("index.html")


# ---------------------------
# 상담 페이지 (대화 유지)
# ---------------------------
@app.route("/chat/<emotion>", methods=["GET", "POST"])
def chat(emotion):
    if "chat_history" not in session:
        session["chat_history"] = []

    chat_history = session["chat_history"]

    if request.method == "POST":
        user_input = request.form["message"]
        chat_history.append({"sender": "user", "text": user_input})

        keyword = detect_keyword(user_input)
        responses = counseling_data[emotion]["keywords"].get(
            keyword, counseling_data[emotion]["keywords"]["기타"]
        )
        bot_reply = random.choice(responses)
        followup = random.choice(counseling_data[emotion]["followups"])

        chat_history.append({"sender": "bot", "text": bot_reply})
        chat_history.append({"sender": "bot", "text": followup})

        session["chat_history"] = chat_history

    return render_template("chat.html", emotion=emotion, chat_history=chat_history)


# ---------------------------
# Flask 서버 실행
# ---------------------------
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
