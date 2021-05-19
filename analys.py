from statistics import Statistics


def main():
    running = True
    while running:
        inp = input('-: ')
        if inp == 'exit':
            running = False
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
