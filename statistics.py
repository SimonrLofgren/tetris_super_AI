import matplotlib.pyplot as plt
from matplotlib import style
from IPython import display
import pickle
import time


class Statistics:
    def __init__(self, episodes, total_score, last_10_scores, times, sum_all_scores,
                 plot_last_10_mean, plot_mean, plot_scores):
        self.episodes = episodes
        self.total_score = total_score
        self.last_10_scores = last_10_scores
        self.times = times
        self.sum_all_scores = sum_all_scores
        self.plot_last_10_mean = plot_last_10_mean
        self.plot_mean = plot_mean
        self.plot_scores = plot_scores

    @staticmethod
    def save_data(data, filename):
        with open(str(filename) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
        print('Data saved.')

    @staticmethod
    def load_data(filename):
        with open(str(filename), 'rb') as f:
            return pickle.load(f), print('Data loaded')

    @staticmethod
    def plot(x, y=0, last_10=0, x2=0, last_10_2=0, xlabel='', ylabel='', title='', ymax=None):
        plt.ion()

        display.clear_output()
        display.display(plt.gcf())
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(x)
        plt.plot(y)
        plt.plot(last_10)
        plt.plot(x2)
        plt.plot(last_10_2)


        plt.ylim(ymin=0, ymax=ymax)

        plt.show(block=True)


    def joakims_plot(self):

        plt.ion()

        display.clear_output()
        display.display(plt.gcf())
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")

        plt.plot(self.plot_scores)
        plt.plot(self.plot_mean)
        plt.plot(self.plot_last_10_mean)
        plt.ylim(ymin=0)

        plt.text(len(self.plot_scores) - 1, self.plot_scores[-1], str(self.plot_scores[-1]))
        plt.text(len(self.plot_mean) - 1, self.plot_mean[-1], str(self.plot_mean[-1]))
        plt.text(len(self.plot_last_10_mean) - 1, self.plot_last_10_mean[-1], str(self.plot_last_10_mean[-1]))
        plt.show(block=False)
        plt.pause(.1)
        plt.savefig("joakims_plot.png")

    def timer(self):
        splits = []
        for t in self.times:
            splits.append(t - self.times[0])
        self.times = []
        return splits[1:len(splits)]

    def t(self):
        t = time.perf_counter()
        self.times.append(t)


    def statistics(self, score, e):

        self.last_10_scores.append(score)
        self.last_10_scores.pop(0)

        self.sum_all_scores += score

        mean_score = self.sum_all_scores / (e)
        last_10_mean = sum(self.last_10_scores) / 10
        self.plot_last_10_mean.append(last_10_mean)
        self.plot_mean.append(mean_score)

        self.plot_scores.append(score)

        self.joakims_plot()
