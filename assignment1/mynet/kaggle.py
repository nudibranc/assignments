import numpy as np
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from mynet import TwoLayerNet
import pandas as pd

#get dataset
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None

#X training data
data = np.array(pd.read_csv('/home/carlos/Downloads/train.csv', sep=',',dtype = np.uint8))
labels = data[:,0]
X_train = data[:, 1:]
print(X_train.shape)
print(labels[0])
imshow(X_train.reshape(-1,28,28)[0])
plt.show()

input_size = 28*28
num_classes = 10
hidden_layers = 300

