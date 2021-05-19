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

    def joakims_plot(self, st):

        plt.ion()

        display.clear_output()
        display.display(plt.gcf())
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Number of Games")
        plt.ylabel("Score")

        plt.plot(st.plot_scores)
        plt.plot(st.plot_mean)
        plt.plot(st.plot_last_10_mean)
        plt.ylim(ymin=0)

        plt.text(len(st.plot_scores) - 1, st.plot_scores[-1], str(st.plot_scores[-1]))
        plt.text(len(st.plot_mean) - 1, st.plot_mean[-1], str(st.plot_mean[-1]))
        plt.text(len(st.plot_last_10_mean) - 1, st.plot_last_10_mean[-1], str(st.plot_last_10_mean[-1]))
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


    def statistics(self, st, score, e):

        st.last_10_scores.append(score)
        st.last_10_scores.pop(0)

        st.sum_all_scores += score

        mean_score = st.sum_all_scores / (e + 1)
        last_10_mean = sum(st.last_10_scores) / 10
        st.plot_last_10_mean.append(last_10_mean)
        st.plot_mean.append(mean_score)

        st.plot_scores.append(score)

        st.joakims_plot(st)
