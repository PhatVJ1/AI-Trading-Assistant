import tensorflow as tf
import numpy as np

def data_generator(data, seq_train, seq_label):
    data = data[['Open', 'Close']]
    data_ai = np.empty((len(data), 0, 2))
    for i in range(seq_train + seq_label - 1, -1, -1):
        data_ai = np.append(data_ai, np.array(data.shift(i))[np.newaxis,:,:].transpose((1, 0, 2)), axis=1)

    data_ai = data_ai[seq_train + seq_label - 1:]

    return data_ai[:, :seq_train, :], tf.transpose(data_ai[:, -seq_label:, :], perm = (0, 2, 1))