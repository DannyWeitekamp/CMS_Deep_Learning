#Get_Data
from keras.datasets import imdb
from keras.preprocessing import sequence
print('Loading data...')
import numpy as np
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
X_train = np.random.random_sample((100,5))
y_train = np.random.random_sample((100,))
X_test = np.random.random_sample((100,5))
y_test = np.random.random_sample((100,))
print(X_train.shape)
print(y_train.shape)
X_train = sequence.pad_sequences(X_train, max_length)
X_test = sequence.pad_sequences(X_test, max_length)
X_train = X_train[:100]
y_train = y_train[:100]
print(X_train.shape)
print(y_train.shape)