
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT


def minimize(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    (thresh, state) = cv2.threshold(state, 0, 255,
                                    cv2.THRESH_BINARY)

    state = np.delete(state, range(0, 96), axis=1)
    state = np.delete(state, range(0, 48), axis=0)
    state = np.delete(state, range(80, 160), axis=1)
    state = np.delete(state, range(160, 192), axis=0)

    return state

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    @property
    def work(self):

        self.env = gym_tetris.make('TetrisA-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        self.env.reset()
        action = np.argmax(self.env.action_space.sample())
        ob, _, _, _ = self.env.step(action)

        inx = 10
        iny = 20
        done = False

        # net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome,
                                                        self.config)
        fitness = 0
        xpos = 0
        xpos_max = 16
        counter = 0
        max_score = 0
        moving = 0

        while not done:
            #self.env.render()
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            # scaledimg = cv2.resize(scaledimg, (iny, inx))
            ob = minimize(ob)
            ob = cv2.resize(ob, (10, 20))

            cv2.imshow('humanwin', scaledimg)
            # cv2.imshow('comwin', ob)
            cv2.waitKey(1)

            imgarray = np.ndarray.flatten(ob)
            # imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)
            action = np.argmax(actions)
            ob, rew, done, info = self.env.step(action)

            xpos = info['score']


            if fitness < xpos:
                fitness = xpos


        print("genome:", self.genome.key, "Fitnes:", fitness)

        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    p = neat.Population(config)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-193')

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(6, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
