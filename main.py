import gzip
import numpy as np
import random

def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 784) / 255.0

def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

X_train = load_images("data/train-images-idx3-ubyte.gz")
y_train = load_labels("data/train-labels-idx1-ubyte.gz")


#print(X_train[0].shape)
#print(y_train[0])
#print(X_train[1][:784])

weights = np.empty([784,10])

weights = np.random.uniform(-1, 1, (784, 10))


for epoch in range(7):
    for dataSet in range(30000):
        outputs = np.dot(X_train[dataSet], weights)
        guess = np.argmax(outputs)

        if guess != y_train[dataSet]:
            weights[:, y_train[dataSet]] += 0.1 * X_train[dataSet]
            weights[:, guess] -= 0.1 * X_train[dataSet]

#Test loop
correct = 0
for dataSet in range(30001,60000):
    outputs = np.dot(X_train[dataSet], weights)
    guess = np.argmax(outputs)

    if guess == y_train[dataSet]:
        correct+=1

correct_percent = correct/(60000-30001) * 100

print (correct_percent, "% Correct")