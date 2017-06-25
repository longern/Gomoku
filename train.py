import keras
from keras.utils import to_categorical
import numpy
from board import Board
from sgf import *

def get_training_data():
	feature = []
	label = []
	for history in read_sgf():
		board = Board()
		for move in history:
			feature.append(numpy.array(board.data))
			choice = numpy.zeros(board.data.shape, int)
			choice[move] = 1
			label.append(choice)
			board.play(move)
	return numpy.array(feature), numpy.array(label)
