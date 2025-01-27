# MAIN AIM : REPRESNT WORDS AS NUMBERS

# use tf keras tokenizer to tokenize the text data
# I - Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
# get the tokenizer from tf.keras.preprocessing.text
from keras_preprocessing.text import Tokenizer

# represent the sentences you want to tokenize as a python array of strings 
# II - Define the sentences
# sentences = [
#     'I love my mug',
#     'I love my dog' 
# ]

# create an instance of the tokenizer object
# [speaking of objects , revise the concept of objects in python]
# num_words : the max number of words that we want to keep
# III - Create an instance of the tokenizer object
# tokenizer = Tokenizer(num_words =100)

# fit_on_texts : this method will fit the tokenizer on the text data
# IV - Fit the tokenizer on the text data
# tokenizer.fit_on_texts(sentences)

# the full list of words is available as word_index property of the tokenizer object
# V - Get the word index
# word_index = tokenizer.word_index

# print the word index
# print(word_index)

# --------------------------------------------
sentences = [
     'I love my mug',
     'I love my dog' ,
     'You drink tea!',
# Should also know how to handle senetences of diff len
     'You love to drink tea from my mug'
]

# tokenizer = Tokenizer(num_words =100)
tokenizer = Tokenizer(num_words =100, oov_token ="<OOV>")
tokenizer.fit_on_texts(sentences) # training
word_index = tokenizer.word_index
# print(word_index)
# {'love': 1, 'my': 2, 'i': 3, 'mug': 4, 'you': 5, 'drink': 6, 'tea': 7, 'dog': 8, 'to': 9, 'from': 10}

# Task 2: Create sequences of tokens from sentences
sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)
# [[3, 1, 2, 4], [3, 1, 2, 8], [5, 6, 7], [5, 1, 9, 6, 7, 10, 2, 4]]


# Now due to above encoding this data is now ready , so that a neural network we created can look at it 
# task 3: how to handle sentences that the neural network has never seen before

test_sentences = [
    "I love my black mug",
    "My dog loves to drink tea"
]


test_sequence = tokenizer.texts_to_sequences(test_sentences)
print(test_sequence)
# Without OOV
# [[3, 1, 2, 4], [2, 8, 9, 6, 7]]  # black is not in the word index so it is ignored , same for loves 
# Hence for above sentences , length of the sequence is not same as the length of the sequence of the training data
# Hence a really big word index will be needed to handle all the words in training and testing data
# How to not loose the seq len : OOV token property [ set it to something you dont expect to see in the corpus ]
# The tokenizer will create a token for OOV and replace all the words it doesn't recognize in the test_sentences and replace it with OOV token. 

# With OOV :  [[4, 2, 3, 1, 5], [3, 9, 1, 10, 7, 8]]

# 
# OOV maintains the sequence len to be the same len as the len of the sentence 
# How does neural network handle sentences of diff len ?
# USE RAGGED TENSORS [ advanced topic ]
# simpler sol : use PADDING 




