import numpy
from keras.utils import to_categorical
from board import Board

channel_size = 22

def get_channels(board: Board, pos: numpy.array):
	directions = numpy.array([(1, -1), (1, 0), (1, 1), (0, 1)])
	channels = numpy.zeros(channel_size)
	channels[0:3] = to_categorical(board.at(pos), 3)
	enemy = lambda x: 3 - x
	for player in [1, 2]:
		max_count = (0, 0)
		for dir in directions:
			count = 0
			liberty = 2
			for i in range(1, 5):
				if not board.in_board(pos + i * dir) or \
					board.data[tuple(pos + i * dir)] != player:
					break
				count += 1
			if not board.in_board(pos + i * dir) or board.at(pos + i * dir) == enemy(player):
				liberty -= 1
			for i in range(-1, -5, -1):
				if not board.in_board(pos + i * dir) or \
					board.data[tuple(pos + i * dir)] != player:
					break
				count += 1
			if not board.in_board(pos + i * dir) or board.at(pos + i * dir) == enemy(player):
				liberty -= 1
			if count >= 5:
				count = 5
			max_count = max(max_count, (count, liberty))
		if player == 1:
			channels[3:9] = to_categorical(max_count[0], 6)
			channels[9:12] = to_categorical(max_count[1], 3)
		else:
			channels[12:18] = to_categorical(max_count[0], 6)
			channels[18:21] = to_categorical(max_count[1], 3)
	channels[21] = board.current_player - 1
	return channels

def get_features(board: Board):
	feature = numpy.zeros((*board.shape, channel_size))
	for pos, _ in numpy.ndenumerate(board.data):
		feature[pos] = get_channels(board, pos)
	return feature

def update_feature(board: Board, feature, move):
	pos_to_update = []
	for dir in numpy.array([(1, -1), (1, 0), (1, 1), (0, 1)]):
		for i in range(1, 6):
			if board.in_board(move + i * dir):
				pos_to_update.append(move + i * dir)
			else:
				break
		for i in range(-1, -6, -1):
			if board.in_board(move + i * dir):
				pos_to_update.append(move + i * dir)
			else:
				break
	pos_to_update.append(move)
	for pos in map(tuple, pos_to_update):
		feature[pos] = get_channels(board, pos)
	for pos, _ in numpy.ndenumerate(board.data):
		feature[pos][21] = board.current_player - 1
