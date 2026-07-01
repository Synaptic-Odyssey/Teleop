#Converts MediaPipe hand landmarks into a clean robotics representation. Encapsulates MediaPipe.


from dataclasses import dataclass
from typing import Optional
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


@dataclass
class HandPose:

    # Wrist position (normalized camera coordinates)
    position: np.ndarray
    
    # 0.0 = fully open, 1.0 = fully closed
    grip: float
    
    confidence: float
    landmarks: NormalizedLandmarkList


class HandPoseExtractor:

    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def extract(
        self,
        hand_landmarks: Optional[NormalizedLandmarkList]
    ) -> Optional[HandPose]:

        if hand_landmarks is None:
            return None

        wrist = hand_landmarks.landmark[self.WRIST]

        position = np.array([
            wrist.x,
            wrist.y,
            wrist.z
        ], dtype=np.float32)

        grip = self.compute_grip(hand_landmarks)

        return HandPose(
            position=position,
            grip=grip,
            confidence=1.0,
            landmarks=hand_landmarks,
        )

    def compute_grip(
        self,
        hand_landmarks: NormalizedLandmarkList,
    ) -> float:

        thumb = hand_landmarks.landmark[self.THUMB_TIP]

        index = hand_landmarks.landmark[self.INDEX_TIP]

        distance = np.sqrt(

            (thumb.x - index.x) ** 2 +

            (thumb.y - index.y) ** 2 +

            (thumb.z - index.z) ** 2

        )

        OPEN_DISTANCE = 0.20

        CLOSED_DISTANCE = 0.02

        grip = 1.0 - (

            (distance - CLOSED_DISTANCE)

            /

            (OPEN_DISTANCE - CLOSED_DISTANCE)

        )

        grip = np.clip(grip, 0.0, 1.0)

        return float(grip)