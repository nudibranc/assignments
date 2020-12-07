import numpy as np
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from twonet import MyTwoNet
import pandas as pd

#get dataset
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None

#predict the data
final_x = np.array(pd.read_csv('/home/carlos/Downloads/test.csv', sep=',',dtype = np.uint8))
print(final_x.shape)

#X training data
data = np.array(pd.read_csv('/home/carlos/Downloads/train.csv', sep=',',dtype = np.uint8))
labels = data[:,0]
X_train = data[:, 1:]

#split for testing
X_test = X_train[40000:,:]
y_test = labels[40000:]
X_train = X_train[:40000,:]
y_train = labels[:40000]
print(X_train.shape)
print(labels.shape)


#initialize hyper hyper parameters
input_size = 28*28
num_classes = 10
hidden_size = 300

#train
mynet = MyTwoNet(input_size,hidden_size,num_classes)
mynet.train(X_train.reshape(-1,28*28),labels,X_train.reshape(-1,28*28),labels, num_iters=10000, verbose= False)

#predict
y_pred = np.argmax(mynet.loss(final_x.reshape(-1,28*28)),axis=1)

#print
for i in range(28000):
    print(f"{str(i+1)},{y_pred[i]}")
imshow(final_x.reshape(-1,28,28)[27999])
plt.show()

