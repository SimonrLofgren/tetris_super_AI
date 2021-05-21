import pandas as pd
import matplotlib.pyplot as plt


def load_csv(file_name):

    df = pd.read_csv(file_name)
    return df


def get_max_value(df):

    print(f"Highest Score: \n{df.loc[df['Total Score'].idxmax()]}")
    print(f"Highest Average: \n{df.loc[df['Average Score'].idxmax()]}")
    print(f"Highest Average Last 10: \n{df.loc[df['Average Score Last 10'].idxmax()]}")


def plot_all_data(df, save=False):

    max_score = df['Total Score'].max()
    x_pos_1 = df[['Total Score']].idxmax()
    max_average = df['Average Score'].max()
    x_pos_2 = df[['Average Score']].idxmax()
    max_a_last = df['Average Score Last 10'].max()
    x_pos_3 = df[['Average Score Last 10']].idxmax()

    ax = plt.gca()
    df.plot(kind='line', x='Episode', y='Total Score', color='black', ax=ax)
    df.plot(kind='line', x='Episode', y='Average Score', color='lime', ax=ax)
    df.plot(kind='line', x='Episode', y='Average Score Last 10', color='red', ax=ax)
    ax.annotate(f'Max Score: {max_score}', xy=(x_pos_1, max_score), xytext=(x_pos_1 + 10, max_score - 50),
                arrowprops=dict(arrowstyle="->", color='black'))
    ax.annotate(f'Max Average: {round(max_average, 2)}', xy=(x_pos_2, max_average), xytext=(x_pos_2, max_average + 300),
                arrowprops=dict(arrowstyle='->', color='lime'))
    ax.annotate(f'Last 10 Max:\n {round(max_a_last, 2)}', xy=(x_pos_3, max_a_last), xytext=(x_pos_3, max_a_last + 200),
                arrowprops=dict(arrowstyle="->", color='red'))

    plt.text(0.901, 0.23, 'Last Scores', color='black',  fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.901, 0.2, (df['Total Score'][df.index[-1]]), color='black', fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.901, 0.17, (df['Average Score'][df.index[-1]]), color='lime', fontsize=8, transform=plt.gcf().transFigure)
    plt.text(0.901, 0.13, (df['Average Score Last 10'][df.index[-1]]), color='red', fontsize=8, transform=plt.gcf().transFigure)

    plt.show()

    if save:
        plt.savefig('Tetris_all_stats.png')


def plot_max_score(df, save=False):

    max_score = df['Total Score'].max()
    x_pos_1 = df[['Total Score']].idxmax()
    last_score = df['Total Score'][df.index[-1]]

    ax = plt.gca()
    df.plot(kind='line', x='Episode', y='Total Score', color='black', ax=ax)
    plt.legend(loc='upper left')
    ax.annotate(f'Max Score: {max_score}', xy=(x_pos_1, max_score), xytext=(x_pos_1 + 10, max_score - 50),
                arrowprops=dict(arrowstyle="->", color='black'))

    plt.text(0.901, 0.2, f'Last Score: \n {last_score}', fontsize=8, transform=plt.gcf().transFigure)

    plt.show()

    if save:
        plt.savefig('Tetris_max_score.png')


def plot_average_score(df, save=False):

    average_score = df['Average Score'].max()
    x_pos_2 = df[['Average Score']].idxmax()
    last_av_score = df['Average Score'][df.index[-1]]

    ax = plt.gca()
    df.plot(kind='line', x='Episode', y='Average Score', color='lime', ax=ax)

    plt.legend(loc='upper left')

    ax.annotate(f'Max Average Score: {round(average_score, 2)}', xy=(x_pos_2, average_score),
                xytext=(x_pos_2 + 10, average_score - 70),
                arrowprops=dict(arrowstyle="->", color='black'))

    plt.text(0.901, 0.2, f'Last Score: \n {round(last_av_score,2)}', fontsize=8, transform=plt.gcf().transFigure)

    plt.show()

    if save:
        plt.savefig('Tetris_average_score.png')


def plot_last_average_score(df, save=False):

    last_average_score = df['Average Score Last 10'].max()
    x_pos_3 = df[['Average Score Last 10']].idxmax()
    last_10_score = df['Average Score Last 10'][df.index[-1]]

    ax = plt.gca()

    df.plot(kind='line', x='Episode', y='Average Score Last 10', color='red', ax=ax)

    plt.legend(loc='upper left')

    ax.annotate(f'Max Last 10 \n Average Score: \n {round(last_average_score, 2)}', xy=(x_pos_3, last_average_score),
                xytext=(x_pos_3 + 10, last_average_score - 50),
                arrowprops=dict(arrowstyle="->", color='black'))

    plt.text(0.901, 0.2, f'Last Score: \n {round(last_10_score,2)}', fontsize=8, transform=plt.gcf().transFigure)

    plt.show()

    if save:
        plt.savefig('Tetris_average_10_score.png')


def main():

    df = load_csv('scores.csv')
    #get_max_value(df)
    #plot_all_data(df, save=False)
    #plot_max_score(df, save=False)
    #plot_last_average_score(df, save=False)


if __name__ == "__main__":
    main()
