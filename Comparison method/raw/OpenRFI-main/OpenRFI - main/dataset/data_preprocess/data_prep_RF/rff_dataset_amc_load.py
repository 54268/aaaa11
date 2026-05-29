import h5py
import numpy as np
import torch.utils.data

from .dataset_load import shuffle_data_with_indices
import os


class Datasets(torch.utils.data.Dataset):


    def get_dataset(self, filename,h5_file_data_set_filename,h5_file_label_filename, h5_file_snr_filename,width,code_mode): 
        DATAS, LABELS, SNRS = loading_dataset_with_directory_for_3_subdatas(filename,
                                                  h5_file_data_set_filename,
                                                  h5_file_label_filename,
                                                  h5_file_snr_filename,
                                                  width,
                                                  code_mode)
        if np.shape(DATAS)[1:] != (1024,2):
            DATAS = np.transpose(DATAS, axes=(0, 2, 1))       

        self.data = DATAS
        self.label = LABELS
        self.snr = SNRS
        self.length = len(self.snr)

        return self 

    def __getitem__(self, item): 
        x = self.data[item, ...]
        y = self.label[item, ...]
        z = self.snr[item, ...]
        return x, y, z

    def __len__(self): 
        return self.length



def loading_dataset_with_directory_for_3_subdatas(directory,
                                                  h5_file_data_set_filename,
                                                  h5_file_label_filename,
                                                  h5_file_snr_filename,
                                                  width,
                                                  code_mode,
                                                  shuffle=False):
    dataset_filenames = os.listdir(directory)
    dataset_filenames = join_file_path(directory, dataset_filenames)
    file_num = len(dataset_filenames)
    DATAS = []
    LABELS = []
    SNRS = []
    for i in range(file_num):
        dataset_file = h5py.File(dataset_filenames[i], 'r')
        
        
        if code_mode == 'debug':
            DATAS.append(dataset_file[h5_file_data_set_filename][:1024, :width])
            LABELS.append(dataset_file[h5_file_label_filename][:1024])
            SNRS.append(dataset_file[h5_file_snr_filename][:1024])
        else:
            DATAS.append(dataset_file[h5_file_data_set_filename][:, :width])
            LABELS.append(dataset_file[h5_file_label_filename][:])
            SNRS.append(dataset_file[h5_file_snr_filename][:])
            if code_mode == 'test':
                shuffle = False

        dataset_file.close()

    
    DATAS = np.vstack(DATAS).astype(np.float32) 
    LABELS = np.vstack(LABELS)
    SNRS = np.vstack(SNRS)

    
    
    if shuffle:
        DATAS, LABELS, SNRS = shuffle_data_with_indices(DATAS, LABELS, SNRS)
    return DATAS, np.squeeze(LABELS), SNRS  


def join_file_path(dataset_dir, dataset_filenames):
    for i in range(len(dataset_filenames)):
        dataset_filenames[i] = dataset_dir + '/' + dataset_filenames[i]

    return dataset_filenames
