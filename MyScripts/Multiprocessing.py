import multiprocessing


class Multiprocessing:

    def __init__(self, df, count):
        self.df = df
        self.cpu = int(multiprocessing.cpu_count())
        self.portion = int(count / self.cpu) + 1
        self.list_df = self.subdivide_df()
        self.processes = [None] * self.cpu

    def subdivide_df(self):
        list_df = []
        portion = self.portion
        start = 0
        i = 0
        while i < self.cpu:

            if start < portion * (self.cpu - 1):
                if start == 0:
                    data = self.df.iloc[start:portion, :]
                else:
                    data = self.df.iloc[start:portion, :]
            else:
                data = self.df.iloc[start:, :]
            start = portion
            portion = portion + self.portion
            list_df.append(data)
            i = i + 1
        return list_df
