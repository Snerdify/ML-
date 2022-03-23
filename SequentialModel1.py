from keras.models import Sequential 
from keras.layers import Dense,Activation


model=keras.Sequential(
    [
    layers.Dense(2,activation='relu'),
    layers.Dense(3,activation='relu'),
    layers.Dense(4),
    ]
)


#how to call the model on a test input
x=tf.ones((3,3))
y = model(x)


#how to access these layers-here model.layers wont show the input object
model.layers

#add the layers
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))


#how to remove the layers
model.pop()
print(len(model.layers))

#the input shape-model needs to know what input shape it should expect
#as the first layer is the sequential layer , it should know which 
#input to recieve
#the layers after that can do automatic shape inference

layer=layers.Dense(3)
layer.weights #here we will get an empty array as we havent passed the input values yet

# weights are created when we call it on an input. Shape of the
#weights depends on shape of the inputs
x=tf.ones((1,4))
y=layers(x)
layer.weights

print("number of weights after calling the model " ,len(model.weights))


model.summary 


# if you wish to see theinput object as well
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()
