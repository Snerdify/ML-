# MAIN AIM : REPRESNT WORDS AS NUMBERS

# use tf keras tokenizer to tokenize the text data
# I - Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
# get the tokenizer from tf.keras.preprocessing.text
from keras_preprocessing.text import Tokenizer

# represent the sentences you want to tokenize as a python array of strings 
# II - Define the sentences
sentences = [
    'I love my mug',
    'I love my dog'
]

# create an instance of the tokenizer object
# [speaking of objects , revise the concept of objects in python]
# num_words : the max number of words that we want to keep
# III - Create an instance of the tokenizer object
tokenizer = Tokenizer(num_words =100)

# fit_on_texts : this method will fit the tokenizer on the text data
# IV - Fit the tokenizer on the text data
tokenizer.fit_on_texts(sentences)

# the full list of words is available as word_index property of the tokenizer object
# V - Get the word index
word_index = tokenizer.word_index

# print the word index
print(word_index)



