import tensorflow as tf
from tensorflow import keras
# get the tokenizer from tensorflow.keras.preprocessing.text
from keras_preprocessing.text import Tokenizer
# Inport added whilst implementing padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
     'I love my mug',
     'I love my dog' ,
     'You drink tea!',
# Should also know how to handle sentences of diff len
     'You love to drink tea from my mug'
]
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
# test_sequence = tokenizer.texts_to_sequences(test_sentences)
# print(test_sequence)
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
padded = pad_sequences(sequences)
print(word_index)
print(sequences)
print(padded)
# {'<OOV>': 1, 'love': 2, 'my': 3, 'i': 4, 'mug': 5, 'you': 6, 'drink': 7, 'tea': 8, 'dog': 9, 'to': 10, 'from': 11}
# [[4, 2, 3, 5], [4, 2, 3, 9], [6, 7, 8], [6, 2, 10, 7, 8, 11, 3, 5]]   
# [[ 0  0  0  0  4  2  3  5]
#  [ 0  0  0  0  4  2  3  9]
#  [ 0  0  0  0  0  6  7  8] 
#  [ 6  2 10  7  8 11  3  5]]

# The padding for the first word : 4,2,3,5 has 4 zeroes preceding it bcoz 
# the longest sentences has 8 words . 
# if we want zeroes after the numerical words: 
# padded = pad_sequences(sequences, padding = 'post')
# If we dont want the pad len to be the same as the len of the longest sequence:
# padded = pad_sequences(sequences, maxlen = 5)


# What if the sentences are longer than the max len specified in the padding , use truncate
# padded = pad_sequences(sequences, padding='post' , truncating='post' , maxlen = 5)


