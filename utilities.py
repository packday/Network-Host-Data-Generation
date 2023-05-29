import numpy as np


def split_training_test_data(dataframe, percentage):
    msk = np.random.rand(len(dataframe)) < percentage
    training = dataframe[msk]
    testing = dataframe[~msk]

    return [training, testing]