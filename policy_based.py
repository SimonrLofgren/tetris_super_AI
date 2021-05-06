import random
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import cv2
from collections import deque
import numpy as np
import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class CustomProcessor(Processor):
    '''
    acts as a coupling mechanism between the agent and the environment
    '''

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        return np.squeeze(batch, axis=1)

def build_model(state, actions):
    model = Sequential()
    # model.add(Flatten(input_shape=(1, state)))
    model.add(Dense(state, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=0.01)
    return dqn


def run_test(env):
    episodes = 10
    for e in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            n_state, reward, done, info = env.step(action)
            score+=reward
        print(f'Episode: {e}, score: {score}')


def main():

    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MOVEMENT)

    state_input_size = env.observation_space.shape[0]
    number_of_actions = env.action_space.n

    # run_test(env)

    model = build_model(state_input_size, number_of_actions)
    model.summary()

    dqn = build_agent(model, number_of_actions)
    dqn.compile(Adam(lr=0.001), metrics=['mae'])
    dqn.fit(env, nb_steps=1000, visualize=False, verbose=1)


if __name__ == "__main__":
    main()
