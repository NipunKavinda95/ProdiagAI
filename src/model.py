import numpy as np

def predict_health(temp, vibration, hours):
    if temp < 60 and vibration < 3:
        return "Normal"
    elif temp < 75 and vibration < 6:
        return "Warning"
    else:
        return "Fault"