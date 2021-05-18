import matplotlib as plt
import

def plot():
    style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim(0, 2)

    plt.plot(iterations, total_times)
    plt.show()
    plt.savefig("iterationTime.png")