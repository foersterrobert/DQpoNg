import cv2
import mediapipe as mp
import random

class Hand:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

    def get_action(self, sucess, image):
        run = True
        if not sucess:
            print("empty camera frame")
            return run, (random.randint(0, 1))
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        action = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
        image = cv2.flip(image, 1)
        
        if results.multi_hand_landmarks:
            if results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[2].y:
                action = -1
                cv2.putText(image, "UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                action = 1
                cv2.putText(image, "DOWN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not action:
            action = random.randint(0,1)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            run = False
        
        return run, action

