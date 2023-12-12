import time
import csv
from statistics import mean

class TrainBenchmark:
    def __init__(self, num_iters):
        self.iter_start = None
        self.curr_iter = -1
        self.num_iters = num_iters
        self.curr_epoch = 0
        self.epoch_times = []
        self.iter_times = []

    def __iter__(self):
        self.iter_start = time.perf_counter()
        return self

    def add_epoch(self, epoch_time):
        self.epoch_times[self.curr_iter].append(epoch_time);

    def __next__(self):
        iter_elapsed = time.perf_counter() - self.iter_start
        self.iter_times.append(iter_elapsed)

        self.curr_iter += 1
        if self.curr_iter >= self.num_iters:
            raise StopIteration

        self.curr_epoch = 0
        self.epoch_times.append([])
        self.iter_start = time.perf_counter()

    def write_to_csv(self, filepath):
        num_epochs = len(self.epoch_times[0])
        headers = [f'epoch_{i}' for i in range(num_epochs)]
        headers.append('avg')
        headers.append('iter_total')


        with open(filepath, 'w+', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(headers)

            for times, total in zip(self.epoch_times, self.iter_times):
                row = times + [mean(times), total]
                csv_writer.writerow(row)

