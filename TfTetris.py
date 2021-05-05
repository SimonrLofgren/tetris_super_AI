import random

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import cv2
from collections import deque
import numpy as np
import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam


class Agent:
    def __init__(self, state_input_size, number_of_actions):
        self.state_input_size = state_input_size
        self.number_of_action = number_of_actions
        self.learning_rate = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9
        self.batch_size = 512
        self.training_start = 1000
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(240, input_dim=self.state_input_size))
        model.add(Dense(self.number_of_action))
        model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))
        return model

    def train_model(self):
        pass

    def test_model(self):
        pass

    def get_action(self, state):
        rand_val = np.random.rand()
        if rand_val <= self.epsilon:
            return random.randrange(self.number_of_action)
        else:
            q_learning_value = self.model.predict(state)
            return np.argmax(q_learning_value[0])

EPISODES = 3000
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
cv2.namedWindow('ComWin', cv2.WINDOW_NORMAL)

state_input_size = env.observation_space.shape[0]
number_of_actions = env.action_space.n
agent = Agent(state_input_size, number_of_actions)
done = True

for e in range(EPISODES):
    score = 0
    if done:
        state = env.reset()
    grayimg = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    #grayimg = np.reshape(grayimg, [256, state_input_size])
    cv2.imshow('ComWin', grayimg)
    env.render()
    action = agent.get_action(grayimg)
    state, reward, done, info = env.step(action)


env.close()



