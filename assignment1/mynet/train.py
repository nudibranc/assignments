import numpy as np
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from mynet import TwoLayerNet

#get dataset
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None

#X training data
with open('train-images-idx3-ubyte',"rb") as f:
    x_raw = np.fromfile(f,dtype = np.uint8)
    X_train =x_raw[0x10:].reshape(-1,28,28)
    X_val = X_train[50000:,:]
    X_train = X_train[:50000,:]

#Y training labels
with open('train-labels-idx1-ubyte',"rb") as f1:
    y_raw = np.fromfile(f1,dtype = np.uint8)[8:]
    y_val = y_raw[50000:]
    y_train = y_raw[:50000]

#X test
with open('t10k-images-idx3-ubyte',"rb") as f:
    X_test = np.fromfile(f,dtype = np.uint8)
    X_test =X_test[0x10:].reshape(-1,28,28)

#Y test
with open('t10k-labels-idx1-ubyte',"rb") as f1:
    y_test = np.fromfile(f1,dtype = np.uint8)[8:]

#2 layered NN
input_size = 28 * 28
hidden_size = 250
num_classes = 10

#initialize the net
mynet = TwoLayerNet(input_size,hidden_size,num_classes)
mynet.train(X_train.reshape(-1,28*28),y_train,X_val.reshape(-1,28*28),y_val)
y_pred = mynet.predict(X_test.reshape(-1,28*28))
print(X_test.shape)
print(y_pred.shape)
print(y_test.shape)

#check accuracy
print(np.mean(y_pred == y_test))
imshow(X_test[10])
print(y_pred[10])
plt.show()

