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
			label = numpy.zeros(board.data.shape, int)
			label[move] = 1
			labels.append(label)
			board.play(move)
	return numpy.array(features), numpy.array(labels)

generate_sgf_data = False
if generate_sgf_data:
	x_train, y_train = get_training_data()
	numpy.save("board.npy", x_train)
	numpy.save("choice.npy", y_train)
else:
	x_train = numpy.load("board.npy")
	y_train = numpy.load("choice.npy")

x_train = numpy.concatenate((x_train, numpy.flip(x_train, 1), numpy.flip(x_train, 2)))
y_train = y_train.reshape((-1, 15, 15))
y_train = numpy.concatenate((y_train, numpy.flip(y_train, 1), numpy.flip(y_train, 2))).reshape((-1, 225))

print("Data Loaded")

load_model = False
if not load_model:
	input1 = keras.layers.Input(shape=(15, 15, channel_size))
	layer = keras.layers.ZeroPadding2D(2)(input1)
	layer = keras.layers.Conv2D(48, (5, 5), activation="relu")(layer)
	layer = keras.layers.ZeroPadding2D(1)(layer)
	layer = keras.layers.Conv2D(48, (3, 3), activation="relu")(layer)
	layer = keras.layers.ZeroPadding2D(1)(layer)
	layer = keras.layers.Conv2D(48, (3, 3), activation="relu")(layer)
	layer = keras.layers.ZeroPadding2D(1)(layer)
	layer = keras.layers.Conv2D(48, (3, 3), activation="relu")(layer)
	layer = keras.layers.ZeroPadding2D(1)(layer)
	layer = keras.layers.Conv2D(48, (3, 3), activation="relu")(layer)
	layer = keras.layers.LocallyConnected2D(1, (1, 1))(layer)
	layer = keras.layers.Reshape((225, ))(layer)
	layer = keras.layers.Activation("softmax")(layer)
	model = keras.models.Model(inputs=input1, outputs=layer)
	model.compile("sgd", keras.losses.categorical_crossentropy, metrics=["acc", "top_k_categorical_accuracy"])
else:
	model = keras.models.load_model("policy.h5")

model.fit(x_train, y_train, epochs=30, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(patience=2)])
model.save("policy.h5")
