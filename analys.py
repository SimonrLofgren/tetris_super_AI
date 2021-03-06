from statistics import Statistics
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

def main():
    running = True
    while running:
        print('1. one')
        print('2. two')
        print('3. mean')
        print('4. Live updates')
        inp = input('-: ')
        if inp == 'exit':
            running = False


        elif inp == '2':
            try:
                inp = input('-: ')
                data = Statistics.load_data(f'statistics_data/statistics{inp}.pkl')
                inp = input('-: ')
                data2 = Statistics.load_data(f'statistics_data/statistics{inp}.pkl')
                data = data[0]
                data2 = data2[0]

                mean_splits = data[0]
                last_10 = data[1]
                mean_splits2 = data2[0]
                last_10_2 = data2[1]

                Statistics.plot(mean_splits, last_10=last_10, x2=mean_splits2, last_10_2=last_10_2, ymax=0.5)
            except:
                print('not valid')

        elif inp == '3':

            try:
                data = Statistics.load_data(f'meanIterTimes.pkl')
                mean = data[0]


                Statistics.plot(mean, ymax=0.5, xlabel='episode', ylabel='Mean_time')
            except:
                print('not valid')

        elif inp == '4':
            print('which graph?')
            print('1. Total score')
            print('2. Average score')
            print('3. Average score last 10')

            inp = input('-: ')

            if inp == '1':
                try:
                    csv = pd.read_csv(r'scores.csv')
                    res = seaborn.lineplot(x="Episode", y="Total Score", data=csv)
                    plt.show()
                except Exception as e:
                    print(e)
                    print('not valid')

            if inp == '2':
                try:
                    csv = pd.read_csv(r'scores.csv')
                    res = seaborn.lineplot(x="Episode", y="Average Score", data=csv)
                    plt.show()
                except Exception as e:
                    print(e)
                    print('not valid')

            if inp == '3':
                try:
                    csv = pd.read_csv(r'scores.csv')
                    res = seaborn.lineplot(x="Episode", y="Average Score Last 10", data=csv)
                    plt.show()
                except Exception as e:
                    print(e)
                    print('not valid')
            else:
                pass

        else:
            try:
                data = Statistics.load_data(f'statistics_data/statistics{inp}.pkl')
                data = data[0]
                mean_splits = data[0]
                last_10 = data[1]
                Statistics.plot(mean_splits, last_10=last_10, ymax=0.5)
            except:
                print('not valid')



if __name__ == "__main__":
    main()
