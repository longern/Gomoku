import keras
from keras.utils import to_categorical
import numpy
import random
from board import Board
from sgf import *

channel_size = 3

def get_channels(board: Board, pos: numpy.array):
	directions = numpy.array([(1, -1), (1, 0), (1, 1), (0, 1)])
	channels = numpy.zeros(channel_size)
	channels[0:3] = to_categorical(board.data[pos], 3)
	return channels
	for player in [1, 2]:
		for dir in directions:
			count = 0
			for i in range(1, 5):
				if not board.in_board(pos + i * dir) or \
					board.data[tuple(pos + i * dir)] != player:
					break
				count += 1
			for i in range(-1, -5, -1):
				if not board.in_board(pos + i * dir) or \
					board.data[tuple(pos + i * dir)] != player:
					break
				count += 1
			if count >= 5:
				count = 5
		if player == 1:
			channels[3:9] = to_categorical(count, 6)
		else:
			channels[9:15] = to_categorical(count, 6)
	return channels

def get_training_data():
	features = []
	turns = []
	labels = []
	for history in random.sample(read_sgf(), 1000):
		board = Board()
		for move in history:
			feature = numpy.zeros((15, 15, channel_size))
			for pos, _ in numpy.ndenumerate(board.data):
				feature[pos] = get_channels(board, pos)
			features.append(feature)
			turns.append(*to_categorical(board.current_player - 1, 2))
			label = numpy.zeros(board.data.shape, int)
			label[move] = 1
			labels.append(label.reshape(225))
			board.play(move)
	return [numpy.array(features), numpy.array(turns)], numpy.array(labels)

x_train, y_train = get_training_data()
numpy.save("board.npy", x_train[0])
numpy.save("choice.npy", y_train)
#x_train = numpy.load("board.npy")
#y_train = numpy.load("choice.npy")

print("Data Loaded")
input1 = keras.layers.Input(shape=(15, 15, channel_size))
input2 = keras.layers.Input(shape=(2,))
layer = keras.layers.concatenate([
	keras.models.Sequential([
		keras.layers.Conv2D(24, (3, 3), activation='relu', input_shape=(15, 15, channel_size)),
		keras.layers.Conv2D(32, (3, 3), activation='relu'),
		keras.layers.Dropout(0.5),
		keras.layers.Flatten(),
	])(input1),
	input2]
)
layer = keras.layers.Dense(1024, activation="relu")(layer)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(225, activation="softmax")(layer)
model = keras.models.Model(inputs=[input1, input2], outputs=layer)

model.compile("sgd", keras.losses.categorical_crossentropy, metrics=["top_k_categorical_accuracy"])

model.fit(x_train, y_train, epochs=50, validation_split=0.1)
model.save("model.h5")
