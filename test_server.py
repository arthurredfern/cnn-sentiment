"""A quick script to test a model running in the TensorFlow Model Server.
It sends a request for each raw sentence in the whole dataset. As such,
it shouldn't be used to evaluate model performance but to check that the model
is being served properly.
"""

import requests
import numpy as np

# We assume the ModelServer is running and exported a REST
# API at port 8501
url = 'http://localhost:8501/v1/models/default:predict'
predictions = []
with open('data/clean/data.txt') as file:
    for line in file:
        # Remove new line character
        data = {"inputs": [line[:-1]]}
        # Request the server for a new prediction
        r = requests.post(url, json=data)
        positive = r.json()['outputs'][0][0]
        predictions.append(1 if positive else 0)

# Evaluate accuracy
truth = np.loadtxt('data/clean/labels.txt', dtype=np.int)
accuracy = np.sum(predictions == truth)/len(predictions)
print('Accuracy: {:.3f}'.format(accuracy))
