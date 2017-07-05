import numpy
from board import Board
from feature import get_features
from model_manager import model_manager

gauss_noise = lambda scale, shape: numpy.random.normal(0, scale, shape)

def ai_move(board: Board, feature = None, policy:str = "max"):
	if board.turn == 1:
		return naive_ai(board)
	if feature is None:
		feature = get_features(board)
	score = lambda x:naive_score(x, board.current_player)
	result = model_manager["reinforce"].predict(numpy.array([feature])).reshape(15, 15)
	if policy != "sample":
		result += numpy.apply_along_axis(score, 2, feature)
	result = result * (board.data == 0)
	result /= result.sum()
	if policy == "max":
		choice = numpy.unravel_index(result.argmax(), board.shape)
	elif policy == "sample":
		choice = numpy.unravel_index(numpy.random.choice(225, p=result.flatten()), board.shape)
	return choice

def random_move(board: Board):
	mask = (board.data == 0)
	return numpy.unravel_index(numpy.random.choice(225, p=mask.flatten()/mask.sum()), board.shape)

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

def naive_ai(board: Board, feature = None):
	if feature is None:
		feature = get_features(board)
	score = lambda x:naive_score(x, board.current_player)
	result = numpy.apply_along_axis(score, 2, feature) + gauss_noise(10e-6, board.shape)
	mask = (board.data == 0)
	return numpy.unravel_index((result * mask).argmax(), board.shape)
