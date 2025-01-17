# Build A neural network without any hidden layers - A perceptron

# Inputs -> Synapse -> Neuron -> Outputs 

# Inputs[Training Data] -> shape(3,4) amtrix - training examples - each row is a training example , there are 4 rows with 3 inputs in each row  

# Populate the weights - x1, x2 ,x3 ...

# Add weights as w1, w2 , w3...  to different synapses - (3,1) matrix - 3 inputs , one output - generate random values from -1 to 1 , with a mean of 0
The weights need to be initialized before the training process, and random initialization ensures that the perceptron doesn't start with all weights being the same, avoiding symmetric learning issues.
# why we are subtracting 1 from synaptic weights : This makes the initial weights distributed across a balanced range of positive and negative values, which helps the learning process. Without this, weights would be initialized with only non-negative values (if we just used np.random.random), potentially causing biased updates in the training process.

# Balanced weights in the range [-1, 1) introduce variety, which prevents the model from learning redundant features or getting stuck in suboptimal configurations.

# If weights are all initialized to the same value (e.g., all 0 or all 1), the gradients for all weights would be the same during backpropagation, and the perceptron might learn the same updates for all weights, effectively not learning properly.

# Random initialization spreads the starting point across the weight space, giving the model a better chance of finding the optimal weights faster.



If all weights started positive (e.g., range [0, 1)), the network could take longer to converge or fail to converge to a good solution because the search space for updates would be skewed.


# Neuron - Calculates Weighted sum of the inputs[ x1w1 + x2w2 +x3w3 ]

# Put the weighted sum through a normalizing function to produce output 
# Normalizing function - Produces an outcome between 0 and 1

# Sigmoid Func -> (1/1+e^-x) -> Apply this to weighted sum to get an outcome between 0 and 1 . This is how to calculate the output 

# Output[Training output] - y - shape(4,1) matrix , one output for 4 rows of training examples 


## TRAINING PROCESS
1. Calculate the output by using sigmoid func on the weighted sum of inputs and weights
2. Cal error by subtracting the calculated output from actual output 
3. Update the weights depending on severeness of the error
4. Repeat the process for 10,000 iterations

# By how much to update the weights
ERROR WEIGHTED DERIVATIVES 
Weights adjust by = error.input.(gradient of sigmoid func[Output])

The adjustment we make to the weights should be proportional to the size of errors.

Its also proportional to outputs
If the output is large that means the weight was heavy , that means the neuron was pretty confident in its prediction. We don't need to adjust that kind of weights 

If the output was small , that means the weights were small and hence the neuron was less confident , adjust these weights more . 

Adjustment is also proportional to the inputs too. Inouts are 0 or 1. 





