import keras
import numpy
from board import Board
from feature import get_features
from keras.utils import to_categorical
model = keras.models.load_model("model.h5")

gauss_noise = lambda scale, shape: numpy.random.normal(0, scale, shape)

def ai_move(board: Board):
	if board.turn == 1:
		return naive_ai(board)
	score = lambda x:naive_score(x, board.current_player)
	result = model.predict([numpy.array([get_features(board)]), numpy.array(to_categorical(board.current_player - 1, 2))]).reshape(15, 15) + \
		numpy.apply_along_axis(score, 2, get_features(board)) + \
		gauss_noise(0.1 * numpy.exp(-board.turn), board.shape)
	mask = (board.data == 0)
	return numpy.unravel_index((result * mask).argmax(), board.shape)

def naive_score(x: numpy.array, player: int):
	enemy = 3 - player
	if not x[0]:
		return 0
	if numpy.argmax(x[player * 9 - 6:player * 9]) == 4:
		return 4
	elif numpy.argmax(x[enemy * 9 - 6:enemy * 9]) == 4:
		return 3
	elif numpy.argmax(x[player * 9 - 6:player * 9]) == 3 and numpy.argmax(x[player * 9:player * 9 + 3]) == 2:
		return 2
	elif numpy.argmax(x[enemy * 9 - 6:enemy * 9]) == 3 and numpy.argmax(x[enemy * 9:enemy * 9 + 3]) == 2:
		return 1
	else:
		return numpy.argmax(x[3:9]) * 10e-4

def naive_ai(board: Board):
	score = lambda x:naive_score(x, board.current_player)
	result = numpy.apply_along_axis(score, 2, get_features(board)) + gauss_noise(10e-6, board.shape)
	mask = (board.data == 0)
	return numpy.unravel_index((result * mask).argmax(), board.shape)
