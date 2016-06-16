#Compile
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=max_length, dropout=0.2))
model.add(LSTM(embedding_dim, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])