import matplotlib.pyplot as plt
# pip install gym-retro
import retro
import numpy as np
import cv2
import neat
import pickle
import os
from visualize import *


env = retro.make(game='SuperMarioWorld-Snes', state='Start')
image_array, game_fitness, genome_generations = [], [], []


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        obs = env.reset()
        action = env.action_space.sample()

        # size of emulator image, resolution 256×224
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_fitness = 0
        frame_count = 0
        counter = 0
        score = 0
        score_tracker = 0
        coins = 0
        coins_tracker = 0
        x_position_pre = 0
        checkpoint = False
        checkpoint_value = 0
        end_of_level = 0
        jump = 0

        done = False

        while not done:

            env.render()
            frame_count += 1
            obs = cv2.resize(obs, (inx, iny))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (inx, iny))
            image_array = np.ndarray.flatten(obs)

            neural_output = net.activate(image_array)

            obs, rew, done, info = env.step(neural_output)

            score = info['score']
            coins = info['coins']
            dead = info['dead']
            x_position = info['x']
            jump = info['jump']
            checkpoint_value = info['checkpoint']
            end_of_level = info['endOfLevel']

            if score > 0:
                if score > score_tracker:
                    current_fitness = (score * 10)
                    score_tracker = score

                # Add to fitness score if mario gets more coins.
            if coins > 0:
                if coins > coins_tracker:
                    current_fitness += (coins - coins_tracker)
                    coins_tracker = coins

                # As mario moves right, reward him slightly.
            if x_position > x_position_pre:
                if jump > 0:
                    current_fitness += 10
                current_fitness += (x_position / 100)
                x_position_pre = x_position
                counter = 0
                # If mario is standing still or going backwards, penalize him slightly.
            else:
                counter += 1
                current_fitness -= 0.1

            # If mario reaches the checkpoint (located at around xpos == 2425) then give him a huge bonus.
            if checkpoint_value == 1 and checkpoint == False:
                current_fitness += 20000
                checkpoint = True

            # If mario reaches the end of the level, award him automatic winner.
            if end_of_level == 1:
                current_fitness += 1000000
                done = True

            # If mario is standing still or going backwards for 1000 frames, end his try.
            if counter == 250 or dead == 0:
                current_fitness -= 50
                done = True
                print(genome_id, current_fitness)
            genome.fitness = current_fitness


def run_winner(config_file, genome_path='winner_s_mario.pkl'):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    eval_genomes(genomes, config)


def run(config_file):

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint(checkpoint name)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    print(stats)
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix='mario-checkpoint-'))

    winner = p.run(eval_genomes, 4)

    print('\nBest genome:\n{!s}'.format(winner))

    with open('winner_mario.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    plot_stats(stats, ylog=False, view=True, filename='mario_avg_fitness.svg')
    plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_file.txt')
    run_winner(config_path)
    #run(config_path)
