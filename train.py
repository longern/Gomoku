import keras
from keras.utils import to_categorical
import numpy
import random
from board import Board
from feature import *
from sgf import *

def get_training_data():
	features = []
	turns = []
	labels = []
	i = 0
	for history in random.sample(numpy.load("sgf.npy").tolist(), 500):
		i += 1
		print(i)
		board = Board()
		for move in history:
			features.append(get_features(board))
			turns.append(*to_categorical(board.current_player - 1, 2))
			label = numpy.zeros(board.data.shape, int)
			label[move] = 1
			labels.append(label)
			board.play(move)
	return [numpy.array(features), numpy.array(turns)], numpy.array(labels)

x_train, y_train = get_training_data()
numpy.save("board.npy", x_train[0])
numpy.save("turn.npy", x_train[1])
numpy.save("choice.npy", y_train)
"""
x_train = [numpy.load("board.npy"), numpy.load("turn.npy")]
y_train = numpy.load("choice.npy")
"""

x_train[0] = numpy.concatenate((x_train[0], numpy.flip(x_train[0], 1), numpy.flip(x_train[0], 2)))
x_train[1] = numpy.concatenate((x_train[1], x_train[1], x_train[1]))
y_train = y_train.reshape((-1, 15, 15))
y_train = numpy.concatenate((y_train, numpy.flip(y_train, 1), numpy.flip(y_train, 2))).reshape((-1, 225))

print("Data Loaded")
input1 = keras.layers.Input(shape=(15, 15, channel_size))
input2 = keras.layers.Input(shape=(2,))
layer = keras.layers.Conv2D(4, (3, 3), activation="relu")(input1)
layer = keras.layers.Dropout(0.25)(layer)
layer = keras.layers.Flatten()(layer)
layer = keras.layers.concatenate([layer, input2])
layer = keras.layers.Dense(225, activation="softmax")(layer)
model = keras.models.Model(inputs=[input1, input2], outputs=layer)

model.compile("sgd", keras.losses.categorical_crossentropy, metrics=["top_k_categorical_accuracy"])

#model = keras.models.load_model("model.h5")
model.fit(x_train, y_train, epochs=50, validation_split=0.1)
model.save("model.h5")
