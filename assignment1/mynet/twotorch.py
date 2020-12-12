from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import torch

def flatten(x):
    
class MyTwoNet():
    def __init__(self, input_size, hidden_size, output_size):
        #store weights and biases in self_params
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    #loss function, returns gradients if y is given
    def forward(self,X):
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))

      
       
        
        


 

