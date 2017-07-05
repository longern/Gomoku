import keras

class ModelManager:
	def __init__(self, **kwargs):
		self.models = {}
		return super().__init__(**kwargs)

	def __getitem__(self, key):
		if not key in self.models:
			self.models[key] = policy_network = keras.models.load_model(str(key) + ".h5")
		return self.models[key]

	def __delitem__(self, key):
		del self.models[key]

model_manager = ModelManager()
