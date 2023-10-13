import numpy as np
import pandas as pd

class get_data():
    def __init__(self, data_path_or_df, seq_train, seq_test):
        try:
            if isinstance(data_path_or_df, str):
                self.data = pd.read_csv(data_path_or_df)
            elif isinstance(data_path_or_df, pd.DataFrame):
                self.data = data_path_or_df
            self.seq_train = seq_train
            self.seq_test = seq_test
            self.train, self.test = self.data_generator(self.data, seq_train, seq_test)
        except:
            print("data_path_or_df must be path to data csv or DataFrame.")

    def set_seq(self, seq_train = None, seq_test = None):
        if seq_train != None or seq_test != None and seq_train != self.seq_train or seq_test != self.seq_test:
            self.seq_train = seq_train
            self.seq_test = seq_test
            self.train, self.test = self.data_generator(self.data, self.seq_train, self.seq_test)
      
    def set_data_path_or_df(self, data_path_or_df):
        try:
            if isinstance(data_path_or_df, str):
                self.data = pd.read_csv(data_path_or_df)
            elif isinstance(data_path_or_df, pd.DataFrame):
                self.data = data_path_or_df
            self.train, self.test = self.data_generator(self.data)
        except:
            print("data_path_or_df must be path to data csv or DataFrame.")

    def get_train_test(self):
        return self.train, self.test

    def data_generator(self, data, seq_train, seq_test):
        data = data[['Open', 'Close', 'Volume']]
        data_ai = np.empty((len(data), 3, 0))
        for i in range(seq_train + seq_test):
            data_ai = np.append(data_ai, np.array(data.shift(i))[np.newaxis,:,:].transpose(1, 2, 0), axis=2)

        data_ai = data_ai[seq_train + seq_test - 1:, :, ::-1]

        return data_ai[:, :, :seq_train], data_ai[:, :-1, -seq_test:]
    