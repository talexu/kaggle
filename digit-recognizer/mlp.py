'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
train = pd.read_csv(r'./.data/train.csv')
x_train = train.iloc[:, 1:].astype('float32') / 255
y_train = train.iloc[:, 0]

print(x_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

test = pd.read_csv(r'./.data/test.csv')
predict = model.predict(test)
s = pd.Series(np.argmax(predict, axis=1))
s.index += 1
df = pd.DataFrame(s, columns=['Label'])
df.to_csv('submission.csv', index=True, header=True)
