import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import Sequential
from keras.layers import Input, Dense
from keras import optimizers
import numpy as np


def main():
    np.random.seed(9)

    model = Sequential()

    # input
    model.add(Input(shape=(3,)))
    # hidden layer
    model.add(Dense(units=4, activation='sigmoid'))
    # output
    model.add(Dense(units=1, activation='sigmoid'))

    print(model.summary())
    print('')

    sgd = optimizers.SGD(learning_rate=1.0)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model.fit(X, y, epochs=1500, verbose=False)

    print(model.predict(X))


if __name__ == '__main__':
    main()
