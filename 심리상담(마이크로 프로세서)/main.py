from face_emotion_detector_pi import get_emotion_from_face
from counseling_bot import counseling_session

if __name__ == "__main__":
    emotion = get_emotion_from_face()
    if emotion:
        counseling_session(initial_emotion=emotion)
    else:
        print("😢 감정을 인식하지 못했습니다. 프로그램을 종료합니다.")
