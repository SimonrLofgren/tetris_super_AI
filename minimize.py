import numpy as np
import cv2

def Minimize(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    (thresh, state) = cv2.threshold(state, 0, 255,
                                    cv2.THRESH_BINARY)

    state = np.delete(state, range(0, 96), axis=1)
    state = np.delete(state, range(0, 48), axis=0)
    state = np.delete(state, range(80, 160), axis=1)
    state = np.delete(state, range(160, 192), axis=0)

    return state
