import random
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import cv2
from collections import deque
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


# from tensorflow.python.keras.optimizers import Adam  #alla andra
from tensorflow.keras.optimizers import Adam  # Henrik


from statistics import Statistics, init_csv
from minimize import Minimize

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

        else:
            self.state_input_size = state_input_size
            self.number_of_actions = number_of_actions
            self.learning_rate = 0.1
            self.epsilon = 1.0  # how much random nes. 1 = 100%, 0 = 0%
            self.epsilon_min = 0.3
            self.epsilon_decay = 0.9999
            self.batch_size = 512
            self.training_start = 1000
            self.discount_factor = 0.99

        self.memory = deque(maxlen=20000)
        self.model = self.build_model()

        if load_model:
            self.model.load_weights("./tetris.h5")

    def build_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=self.state_input_size,
                        activation='relu'))  # State is input
        model.add(Dense(60, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.number_of_actions,
                        activation='linear'))  # Q_Value of each action is Output
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.number_of_actions)
        else:
            if state.ndim == 1:
                state = np.array([state])
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <state,action,reward,nest_state> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=10, verbose=0)


if __name__ == '__main__':

    SPLIT_THRESH = 0.6
    EPISODES = 3000
    save_point = 0
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print(SIMPLE_MOVEMENT)
    cv2.namedWindow('ComWin',
                    cv2.WINDOW_NORMAL)  # make so the computer window is resizable
    env.reset()

    # get size of state and action from environment
    state_input_size = 200  # adjusted computer input size
    number_of_actions = env.action_space.n

    agent = Agent(state_input_size, number_of_actions)

    st = Statistics([], [], [0] * 10, [], 0, [], [], [])
    Statistics.save_data([], filename='meanIterTimes')
    init_csv()
    for e in range(1, EPISODES):
        st.t()  # Statistics Time

        done = False
        score = 0
        state = env.reset()
        state = Minimize(state)
        state = np.ndarray.flatten(state)
        sum_splits = 0
        counter = 1
        mean_splits = []
        DATA_SAVE = 0
        r = random.randrange(1, 10000)
        frames = 0
        action = agent.get_action(state)
        last_sum_states = [0, 0, 0]
        last_new_block = 0
        clear_line = 0

        while not done:
            st.t()
            if frames == 15:
                env.render()

            # get action for the current state and go one step in environment

            if frames == 15:
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                frames = 0
            else:
                if action == 5:
                    next_state, reward, done, info = env.step(action)
                else:
                    next_state, reward, done, info = env.step(0)
                frames += 1

            next_state = Minimize(next_state)
            cv2.imshow('ComWin', next_state)  # render computer window
            score_state = next_state
            next_state = np.ndarray.flatten(
                next_state)  # flatten 10 by 20 to 1 by 200
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)



            state = next_state

            new_block = info['statistics']
            sum_states = [sum(score_state[-1]), sum(score_state[-2]), sum(score_state[-3])]
            if new_block != last_new_block:
                for i, sum_state in enumerate(sum_states):
                    if last_sum_states[i] < sum_state:
                        reward += (20-i*10) * int((sum_state-last_sum_states[i])/255)
                        last_sum_states[i] = sum_state
            last_new_block = new_block

            if clear_line < int(info['number_of_lines']):
                reward += 1000
                clear_line = int(info['number_of_lines'])

            if frames == 15:
                # every time step do the training
                agent.train_model()
            score += reward

            if done:
                st.total_score.append(score)
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                """mean_iter_times = Statistics.load_data(filename='meanIterTimes.pkl')
                mean_iter_times = mean_iter_times[0]
                mean_iter_times.append(sum_splits/counter)
                Statistics.plot(mean_iter_times, ymax=0.5, xlabel='episode', ylabel='Mean_time')
                Statistics.save_data(mean_iter_times, filename='meanIterTimes')
                mean_iter_times = 0"""

                st.statistics(score, e)

            if (save_point < info['score']) & (load_model is False):
                save_point = info['score']
                agent.model.save_weights(f"tetris.h5")

            """st.t()
            splits = st.timer()
            [print(f'split {i + 1}: {splits[i]}') for i in range(len(splits))
             if splits[i] > SPLIT_THRESH]

            sum_splits += splits[0]
            mean_split = sum_splits / counter
            mean_splits.append(mean_split)

            counter += 1"""

    env.close()
