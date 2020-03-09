
from keras.models import load_model
import numpy as np

def DetectPigy(model,frame):
    x=[]
    x.append([frame])
    Y_pred = model.predict(x)
    y_pred = np.argmax(Y_pred, axis=1)
    return y_pred







