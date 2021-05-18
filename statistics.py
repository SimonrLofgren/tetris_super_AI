import matplotlib.pyplot as plt
from matplotlib import style
from IPython import display
import pickle
import time

class Statistics:
    def __init__(self, episodes, total_score, last_10_scores, times):
        self.episodes = episodes
        self.total_score = total_score
        self.last_10_scores = last_10_scores
        self.times = times

    def save_data(self, data, filename):
        with open(str(filename) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
        print('Data saved.')

    def load_data(self, data):
        with open(str(data)) as f:
            return pickle.load(f), print('Data loaded')

    def plot(self, x, y):
        style.use('fivethirtyeight')
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylim(0, 2)

        plt.plot(x, y)
        plt.show()
        plt.savefig("iterationTime.png")

    def joakims_plot(self, scores, mean_scores, last_10):
        plt.ion()

        display.clear_output()
        display.display(plt.gcf())
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")

        plt.plot(scores)
        plt.plot(mean_scores)
        plt.plot(last_10)
        plt.ylim(ymin=0)

        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.text(len(last_10) - 1, last_10[-1], str(last_10[-1]))
        plt.show(block=False)
        plt.pause(.1)
        plt.savefig("joakims_plot.png")

    def timer(self):
        splits = []
        for t in self.times:
            splits.append(t - self.times[0])
        self.times = []
        return splits

    def t(self):
        t = time.perf_counter()
        self.times.append(t)

