import numpy as np
import pandas as pd
import h5py
import os


def shuffle_data_with_indices(training_data, modulation_type_labels, snr_labels):
    
    data_num = np.shape(training_data)[0]
    shuffle_index = np.random.choice(np.arange(data_num), size=data_num, replace=False)

    shuffle_training_data = training_data[shuffle_index]
    shuffle_modulation_type_labels = modulation_type_labels[shuffle_index]
    shuffle_snr_labels = snr_labels[shuffle_index]

    return shuffle_training_data, shuffle_modulation_type_labels, shuffle_snr_labels
