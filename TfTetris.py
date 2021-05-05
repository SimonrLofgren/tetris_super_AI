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
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(self, state_input_size, number_of_actions):
        self.state_input_size = state_input_size
        self.number_of_action = number_of_actions
        self.learning_rate = 0.1
        self.epsilon = 0.1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9
        self.batch_size = 512
        self.training_start = 1000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(1, input_dim=self.state_input_size))
        model.add(Dense(30720))
        model.add(Dense(60))
        model.add(Dense(24))
        model.add(Dense(self.number_of_action))
        model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def train_model(self):

        if len(self.memory) < self.training_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_input_size))
        update_target = np.zeros((batch_size, self.state_input_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3][0]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))


    def test_model(self):
        pass

    def get_action(self, state):
        rand_val = np.random.rand()
        if rand_val <= self.epsilon:
            return random.randrange(self.number_of_action)
        else:
            if state.ndim == 1:
                state = np.array([state])
            q_learning_value = self.model.predict(state)
            print(np.argmax(q_learning_value[0]))
            return np.argmax(q_learning_value[0])

EPISODES = 3000
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
cv2.namedWindow('ComWin', cv2.WINDOW_NORMAL)

state_input_size = env.observation_space.shape[0] * env.observation_space.shape[1]
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
    grayimg = np.ndarray.flatten(grayimg)
    env.render()
    action = agent.get_action(grayimg)
    state, reward, done, info = env.step(action)


env.close()



