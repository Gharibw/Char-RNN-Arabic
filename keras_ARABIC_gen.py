# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

# importing the training data
# read the local text file.. exists in the same repo
with open('Clean_AA_tweets.txt') as f:
    raw_text = f.read()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)

# create the range!
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# scale the features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
scaledX = sc.fit_transform(dataX) #ADD THE TRAINING SET IN NUMBERS!

# reshape X to be [samples:batch size, time steps, features(predictor)]
X = np.reshape(scaledX, (n_patterns, seq_length, 1))

# one hot encode the next character
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-ARABIC-{epoch:02d}-{loss:.4f}-Keras.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list, initial_epoch=40)

#- ----------------------------------------------------------------------------
# load the network weights
filename = "weights-ARABIC-45-1.3534-Keras.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# generating new characters
def sample_prediction(prediction):
    '''Get rand index from preds based on its prob distribution.

    Params
    ——
    prediction (array (array)): array of length 1 containing array of probs that sums to 1

    Returns
    ——-
    rnd_idx (int): random index from prediction[0]

    Notes
    —–
    Helps to solve problem of repeated outputs.

    len(prediction) = 1
    len(prediction[0]) >> 1
    '''
    X = prediction[0] # sum(X) is approx 1
    rnd_idx = np.random.choice(len(X), p=X)
    return rnd_idx

start = np.random.randint(1, len(dataX)-1)
pattern = dataX[start]
print("Seed: ")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    #index = np.argmax(prediction)
    # per Gustavo’s suggestion, we should not use argmax here
    index = sample_prediction(prediction)
    result = int_to_char[index]
    #seq_in = [int_to_char[value] for value in pattern]
    # not sure why seq_in was here
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print('\nDone.')


