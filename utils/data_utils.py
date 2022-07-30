# Laboratory of Robotics and Cognitive Science
# Version by:  Rafael Anicet Zanini
# Github:      https://github.com/larocs/EMG-GAN

import numpy as np
import pandas as pd
from utils.pickled_data_utilities import load_pickled_data, get_train_val_split_for_fold
#from sklearn.preprocessing import MinMaxScaler

class DataLoader():
    def __init__(self, args):
        self.noise_dim = args.noise_dim
        self.batch_size = args.batch_size
        self.df = load_pickled_data(args.data_dir, args.dataset)
        self.df = self.df[self.df['class'] == args.finger] # filter by finger
        self.train_data = None

    def load_fold(self, fold: int):
        def normalize_seq(seq: np.array) -> np.array:
            min_val = np.min(seq)
            max_val = np.max(seq)
            return (seq - min_val) / (max_val - min_val), min_val, max_val

        # TODO: we need to cache the normalized values
        # TODO: we should train and generate samples in one go maybe?? or write a dataframe / np.arrays with samples and norm values for generation

        train_data, train_lbls, val_data, val_lbls = get_train_val_split_for_fold(self.df, fold)
        #train_data = np.vectorize(normalize_seq, signature='(n)->(k)')(train_data)

        data_norms = np.empty((train_data.shape[0], 2))
        for i in range(train_data.shape[0]):
            train_data[i], data_norms[0], data_norms[1] = normalize_seq(train_data[0])

        self.train_data = np.expand_dims(train_data, axis=2)
        self.data_norms = data_norms

    def shuffle(self):
        np.random.shuffle(self.train_data)

    def get_batches(self):
        num_batches = int(np.ceil(self.train_data.shape[0] / self.batch_size))
        for batch_idx in range(num_batches):
            batch_start_idx = self.batch_size * batch_idx
            if batch_idx < num_batches - 1:
                yield self.train_data[batch_start_idx : self.batch_size * (batch_idx + 1)]
            else:
                yield self.train_data[batch_start_idx:]

    def get_target_seq_len(self):
        return np.vstack(self.df['padded_data'].to_numpy()).shape[1]

# class DataLoader():
#     def __init__(self, args):
#         self.file_path = args['training_file']
#         self.features = args['features']
#         self.channels = len(self.features)
#         self.rescale = args['rescale']
#         self.num_steps = args['num_steps']
#         self.train_split = args['train_split']
#         self.batch_size = args['batch_size']
#
#     def load_training_data(self):
#
#         data = self.load_timeseries(self.file_path, self.features)
#
#         #Normalize data before hand
#         values = data.values
#
#         if self.rescale:
#             values, scalers = self.min_max(values,-1.0,1.0)
#         else:
#             values, scalers = self.normalize(values)
#
#         #Get moving windows samples
#         X_windows = self.get_windows(values,self.num_steps)
#         data = np.array(X_windows)
#         filter_size = round(data.shape[0] * self.train_split)
#         data = data[0:filter_size]
#
#         return data
#
#     def load_timeseries(self, filename, series):
#         #Load time series dataset
#         loaded_series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
#
#         #Applying filter on the selected series
#         selected_series = loaded_series.filter(items=series)
#
#         return selected_series
#
#     def min_max(self, data, min, max):
#         """Normalize data"""
#         scaler = MinMaxScaler(feature_range=(min, max),copy=True)
#         scaler.fit(data)
#         norm_value = scaler.transform(data)
#         return [norm_value, scaler]
#
#     def get_windows(self, data, window_size):
#         # Split data into windows
#         raw = []
#         for index in range(len(data) - window_size):
#             raw.append(data[index: index + window_size])
#         return raw
#
#     def normalize(self, data):
#         """Normalize data"""
#         scaler = MinMaxScaler(feature_range=(0, 1),copy=True)
#         scaler.fit(data)
#         norm_value = scaler.transform(data)
#         return [norm_value, scaler]
#
#     def get_training_batch(self):
#         x_train = self.load_training_data()
#         idx = np.random.randint(0, x_train.shape[0], self.batch_size)
#         signals = x_train[idx]
#         signals = np.reshape(signals, (signals.shape[0],signals.shape[1],self.channels))
#         return signals
