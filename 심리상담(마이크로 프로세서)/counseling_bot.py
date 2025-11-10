import json
import random

def load_counseling_data(file_path="counseling_data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

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

def counseling_session(initial_emotion=None):
    counseling_data = load_counseling_data()
    print("\n상담봇 🤖: 안녕하세요. 잠시 이야기 나눠볼까요?\n")

    emotion = initial_emotion

    while True:
        if not emotion:
            emotion = input("현재 감정을 입력해주세요 (행복/슬픔/화남/놀람) 또는 '끝': ").strip()

        if emotion == "끝":
            print("\n상담봇 🤖: 오늘 이야기 나눠줘서 고마워요 🌿")
            break

        if emotion not in counseling_data:
            print("상담봇 🤖: 아직 그 감정은 다루지 못하지만, 곧 추가될 거예요.\n")
            emotion = None
            continue

        print(f"\n상담봇 🤖: {random.choice(counseling_data[emotion]['intro'])}")

        for _ in range(random.randint(2, 3)):
            user_input = input("당신: ")
            keyword = detect_keyword(user_input)
            response = random.choice(
                counseling_data[emotion]["keywords"].get(keyword, counseling_data[emotion]["keywords"]["기타"])
            )
            print(f"상담봇 🤖: {response}")
            followup = random.choice(counseling_data[emotion]["followups"])
            print(f"상담봇 🤖: {followup}")
            print("────────────────────────────")

        emotion = None
