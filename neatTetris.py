import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
from minimize import Minimize
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT


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
        ob, _, _, _ = self.env.step(int(action))

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
        frames = 0

        while not done:
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            ob = Minimize(ob)
            ob = cv2.resize(ob, (10, 20))

            cv2.imshow('humanwin', scaledimg)
            cv2.waitKey(1)

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)
            action = np.argmax(actions)
            ob, rew, done, info = self.env.step(int(action))

            frames += 1
            if frames == 1200:
                fitness += 1
                frames = 0

        print(
            f"genome:{self.genome.key} Fitnes: {fitness} lines: {info['number_of_lines']}")

        return int(fitness)


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work


def main():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    p = neat.Population(config)

    p = neat.Checkpointer.restore_checkpoint(
        'neat-checkpoint-110')  # load checkpoint

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))  # save checkpoint every "x" time
    cores = 6  # multiprocessing cores
    pe = neat.ParallelEvaluator(cores, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    main()
