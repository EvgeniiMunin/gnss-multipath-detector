from keras import layers
from keras import models
from keras import optimizers

class Model():
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
       	self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))


class Model10():
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
        
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(32, (3,3), activation='relu'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

class Model8():
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        self.model.add(layers.Conv2D(16, (3,3), activation='relu'))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

class Model4():
    def __init__(self, shape):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=shape))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
