# Basic inference code for Traffic Light Agent
# By sachin-iitd

import numpy as np
import os
from keras.models import load_model
import time
class RLAgent:

    def __init__(self, model_name):
        self.qNet = load_model(model_name+'.h5')
        print("Model loaded")

    def getAction(self, state):
        state = [np.array(list(s)).reshape(1,len(s)) for s in state]
        qVal = self.qNet.predict(state)
        return np.argmax(qVal[0])

def test(model, state):
    starttime=time.time()
    action = RLAgent(model).getAction(state)
    endtime=time.time()
    print("Total time is", endtime - starttime)
    print('Action =', action, '@ state =',state, 'for model', model)

if __name__ == "__main__":
    # CPU only inference
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    test('PL',[[1,1,0,0,0,0,0,0],[np.random.randint(10) for _ in range(36)],[0 for _ in range(36)]])
    test('FL-L',[[np.random.random() for _ in range(12)]])
    test('FL-A',[[np.random.random() for _ in range(4)]])
    test('FL-G',[[np.random.random() for _ in range(2)]])
    test('FL-R',[[np.random.random()]])
