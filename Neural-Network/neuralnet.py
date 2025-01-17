import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


training_inputs = np.array([[0,0,1],
                   [1,1,1],
                   [1,0,1],
                   [0,1,1]])

training_output = np.array([[0,1,1,0]]).T

# introduce seed to get the similar outputs every time
np.random.seed(1)


# Randomly distributed,
# Balanced between positive and negative values,
# Diverse enough to allow efficient learning.
synaptic_weights = 2 * np.random.random((3,1)) -1

print("Starting synaptic weights randomly:")
print(synaptic_weights)

for iteration in range(1):
    inputs = training_inputs 
    outputs = sigmoid(np.dot(inputs,synaptic_weights))

print('Outputs after training:')
print(outputs)

# the outputs are wrong because we have initialized the weights randomly
# To get the right outputs , we need a training




