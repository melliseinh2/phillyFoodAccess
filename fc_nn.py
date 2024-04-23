"""
Fully connected neural network architecture. From Lab
Author: Mia and Charlie
Date: April 3rd 2024
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

##################

class FCmodel(Model):
    """
    A fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(FCmodel, self).__init__()

        self.f1 = Flatten()

        self.d1 = Dense(4000, activation = 'relu')
        self.d2 = Dense(1, activation = 'sigmoid')


    def call(self, x):
        unraveled = self.f1(x)
        layer1 = self.d1(unraveled)
        layer2 = self.d2(layer1)

        return layer2 

def two_layer_fc_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    fc_model = FCmodel()

    # shape is: number of examples (mini-batch size), width, height, depth
    #x_np = np.zeros((64, 32, 32, 3))
    x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = fc_model.call(x_np)
    print(output)

    for v in fc_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test two layer function
    two_layer_fc_test()

if __name__ == "__main__":
    main()