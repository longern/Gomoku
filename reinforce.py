import keras
import numpy
from board import Board
from gomoku_ai import *
from feature import update_feature

def reinforce_move(board: Board, feature = None, policy:str = "max"):
	global model
	if feature is None:
		feature = get_features(board)
	score = lambda x:naive_score(x, board.current_player)
	raw_result = model.predict(numpy.array([feature])).reshape(15, 15)
	result = raw_result * (board.data == 0)
	result /= result.sum()
	if policy == "max":
		choice = numpy.unravel_index(result.argmax(), board.shape)
	elif policy == "sample":
		choice = numpy.unravel_index(numpy.random.choice(225, p=result.flatten()), board.shape)
	return choice, raw_result

def optimizer(model, learning_rate=1e-6):
	from keras import backend as K
	action = K.placeholder(shape=[None, 225])
	discounted_rewards = K.placeholder(shape=[None, ])
	
	# Calculate cross entropy error function
	action_prob = K.sum(action * model.output, axis=1)
	cross_entropy = K.log(action_prob) * discounted_rewards
	loss = -K.sum(cross_entropy)
	
	# create training function
	optimizer = keras.optimizers.Adam(lr=learning_rate)
	updates = optimizer.get_updates(model.trainable_weights, [], loss)
	train = K.function([model.input, action, discounted_rewards], [], updates=updates)
	
	return train

model = keras.models.load_model("reinforce.h5")
print("Model loaded")
opt = optimizer(model)
for k in range(1000):
	features = []
	scores = []
	moves = []
	for i in range(200):
		data_start = len(features)
		board = Board()
		first = numpy.random.choice(2) + 1
		feature = get_features(board)
		while board.winner == 0:
			if board.turn <= 2:
				move = random_move(board)
			if board.current_player == first:
				move, result = reinforce_move(board, feature, "sample")
				scores.append(1)
				features.append(feature.copy())
				moves.append(numpy.ravel_multi_index(move, result.shape))
			else:
				move = ai_move(board, feature, "sample")
			board.play(move)
			update_feature(board, feature, move)
		if board.winner == first:
			for j in range(data_start, len(scores)):
				scores[j] *= 0.99 ** (len(scores) - j)
		elif board.winner == 3 - first:
			for j in range(data_start, len(scores)):
				scores[j] *= -0.99 ** (len(scores) - j)
		else:
			pass
	features = numpy.array(features)
	scores = numpy.array(scores)
	scores -= scores.mean()
	scores /= scores.std()
	opt([features, keras.utils.to_categorical(moves, 225), scores])
	if k % 15 == 14:
		model.save("reinforce.h5")

keras.backend.clear_session()
