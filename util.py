import numpy as np
def normalize(sound: np.ndarray):
    return sound / sound.max() if sound.max() != 0 else sound