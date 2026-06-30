from dataclasses import dataclass
import numpy as np


@dataclass
class HandPose:

    #High-level representation of the user's hand. 


    position: np.ndarray       # x,y,z
    rotation: np.ndarray       # quaternion x,y,z,w
    grip: float                # 0=open 1=closed
    confidence: float