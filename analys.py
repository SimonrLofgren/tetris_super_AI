from statistics import Statistics


def main():
    running = True
    while running:
        print('1. one')
        print('2. two')
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

                Statistics.plot(mean_splits, last_10, x2=mean_splits2, last_10_2=last_10_2, ymax=0.5)
            except:
                print('not valid')

        else:
            try:
                data = Statistics.load_data(f'statistics_data/statistics{inp}.pkl')
                data = data[0]
                mean_splits = data[0]
                last_10 = data[1]
                Statistics.plot(mean_splits, last_10, ymax=0.5)
            except:
                print('not valid')



if __name__ == "__main__":
    main()
