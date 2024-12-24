import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
import os
import load_mnist

mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = load_mnist.load()
mnist_data={}
mnist_data['X_train'] = mnist_X_train
mnist_data['y_train'] = mnist_y_train
mnist_data['X_val'] = mnist_X_test
mnist_data['y_val'] = mnist_y_test
mnist_data['X_test'] = mnist_X_test
mnist_data['y_test'] = mnist_y_test
model = FullyConnectedNet(
[300, 300, 300, 300, 300],
weight_scale=3e-2,
input_dim = 784
)
solver = Solver(
model,
mnist_data,
num_epochs=50,
batch_size=100,
update_rule="adam",
optim_config={'learning_rate': 5.5e-4},
verbose=True, lr_decay = 0.9
)
solver.train()
print()
import pickle
with open('testmodel.pickle', 'wb') as handle:
    pickle.dump(model, handle)
