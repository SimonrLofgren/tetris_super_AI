from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import cv2
import numpy as np

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
cv2.namedWindow('ComWin', cv2.WINDOW_NORMAL)
done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    grayimg = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    cv2.imshow('ComWin', grayimg)
    env.render()

env.close()