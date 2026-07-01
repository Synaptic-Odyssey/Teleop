
#Responsible for: opening the webcam, running MediaPipe Hands, returning the raw hand landmarks


import cv2
import mediapipe as mp


class HandTracker:

    def __init__(
        self,
        camera_index: int = 0,
        max_num_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.drawer = mp.solutions.drawing_utils

    def get_frame(self):

        success, frame = self.cap.read()

        if not success:
            return None

        frame = cv2.flip(frame, 1)

        return frame

    def detect(self):

        frame = self.get_frame()

        if frame is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return frame, None

        hand = results.multi_hand_landmarks[0]

        return frame, hand

    def draw(self, frame, hand_landmarks):

        if hand_landmarks is not None:

            self.drawer.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,)

        return frame

    def release(self):

        self.cap.release()
        cv2.destroyAllWindows()