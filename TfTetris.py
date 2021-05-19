import random
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import matplotlib.pyplot as plt
import cv2
from collections import deque
import random
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


load_model = False


class Agent:
    def __init__(self, state_input_size, number_of_actions):
        if load_model:
            self.state_input_size = state_input_size
            self.number_of_actions = number_of_actions
            self.learning_rate = 0.1
            self.epsilon = 0.1
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.9
            self.batch_size = 512
            self.training_start = 1000
            self.discount_factor = 0.99
            self.memory = deque(maxlen=2000)
            self.model = self.build_model()
        else:
            self.state_input_size = state_input_size
            self.number_of_actions = number_of_actions
            self.learning_rate = 0.1
            self.epsilon = 1.0
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.9
            self.batch_size = 512
            self.training_start = 1000
            self.discount_factor = 0.99

        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

        if load_model:
            self.model.load_weights("./tetris.h5")

    def build_model(self):
        model = Sequential()

        model.add(Dense(1, input_dim=state_input_size))
        model.add(Dense(100))
        model.add(Dense(24))
        model.add(Dense(self.number_of_actions))
        model.summary()
        model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

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
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def test_model(self):
        pass

    def get_action(self, state):
        rand_val = np.random.rand()
        if rand_val <= self.epsilon:
            return random.randrange(self.number_of_actions)
        else:
            if state.ndim == 1:
                state = np.array([state])
            q_learning_value = self.model.predict(state)
            return np.argmax(q_learning_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def minimize(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    (thresh, state) = cv2.threshold(state, 0, 255,
                                      cv2.THRESH_BINARY)

    state = np.delete(state, range(0, 96), axis=1)
    state = np.delete(state, range(0, 48), axis=0)
    state = np.delete(state, range(80, 160), axis=1)
    state = np.delete(state, range(160, 192), axis=0)
    state = cv2.resize(state, (inx, iny))
    return state


episodes, scores = [], []
EPISODES = 3000
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
cv2.namedWindow('ComWin', cv2.WINDOW_NORMAL)

state_input_size = int(200)
number_of_actions = env.action_space.n
agent = Agent(state_input_size, number_of_actions)
done = True

inx = 10
iny = 20
for e in range(1, EPISODES):
    done = False
    save_point = 0
    score = 0
    max_score = 0
    state = env.reset()
    board_height = 0
    state = np.ndarray.flatten(minimize(state))
    env.reset()

    while not done:

        env.render()

        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = minimize(next_state)
        cleared_lines = 0
        score = 0

        cv2.imshow('ComWin', next_state)
        next_state = np.ndarray.flatten(next_state)

        score += reward

        if score < info['score']:
            score = info['score']
            reward += score + 1

        if info['board_height'] > board_height:
            reward = reward - 1
            board_height = info['board_height']

        if cleared_lines < info['number_of_lines']:
            cleared_lines = info['number_of_lines']
            reward += score * 2

        agent.append_sample(state, action, reward, next_state, done)
        agent.train_model()
        state = next_state

        if done:
            reward = reward - 100
            scores.append(score)
            episodes.append(e)
            plt.plot(episodes, scores, 'b')
            plt.savefig("tetris_round.png")
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon)

        if (save_point < info['score']) & (load_model == False):
            save_point = info['score']
            agent.model.save_weights("tetris.h5")

    line = info['number_of_lines']
    print(f'Episode:{e}  Score:{score} Lines:{line} epsilon:{agent.epsilon}')

env.close()
