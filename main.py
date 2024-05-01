import cv2
import mediapipe as mp


def detect_victory_gesture(frame, hand_landmarks):
    if hand_landmarks is not None:
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

        if index_finger_tip.y < middle_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y and ring_finger_tip.y < pinky_tip.y:
            return True

    return False


def main():
    cap = cv2.VideoCapture(0)  
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)

        
        hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

        
        victory_gesture = detect_victory_gesture(frame, hand_landmarks)

        
        if victory_gesture:
            cv2.putText(frame, "Victory Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        cv2.imshow('Frame', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

