from .rff_dataset_amc_load import Datasets
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

dataset_dir_name = '2022_12/steady_signal'

multiple_files_training_dataset_dir = '../data/' + dataset_dir_name + '/training'

multiple_files_valid_dataset_dir = '../data/' + dataset_dir_name + '/valid'
multiple_files_test_dataset_dir = '../data/' + dataset_dir_name + '/valid'


h5_file_data_set_filename = 'noisy_datas'
h5_file_label_filename = 'labels'
h5_file_snr_filename = 'snrs'
sample_size_dim = 0


training_dataset = Datasets().get_dataset(filename=multiple_files_training_dataset_dir,h5_file_data_set_filename='noisy_datas',h5_file_label_filename='labels', h5_file_snr_filename='snrs',width=1024,code_mode='training')
valid_dataset = Datasets().get_dataset(filename=multiple_files_valid_dataset_dir,h5_file_data_set_filename='noisy_datas',h5_file_label_filename='labels', h5_file_snr_filename='snrs',width=1024,code_mode='training')
testing_dataset = Datasets().get_dataset(filename=multiple_files_test_dataset_dir,h5_file_data_set_filename='noisy_datas',h5_file_label_filename='labels', h5_file_snr_filename='snrs',width=1024,code_mode='training')


def get_labeled_index( labeled_classes, labeled_ratio,targets):

    labeled_idxs = []
    unlabeled_idxs = []
    rand_number = 0
    np.random.seed(rand_number)
    for idx, label in enumerate(targets):
        if label in labeled_classes and np.random.rand() < labeled_ratio:
            labeled_idxs.append(idx)
        else:
            unlabeled_idxs.append(idx)
    return labeled_idxs, unlabeled_idxs


def shrink_data( idxs,targets,data):
    targets = np.array(targets)
    targets = targets[
        idxs].tolist()
    data = data[
        idxs, ...]
    return data,targets

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
               "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]
labeled_num=24
labeled_classes = range(labeled_num)

labeled_idxs, unlabeled_idxs = get_labeled_index(labeled_classes, labeled_ratio=0.8,targets=training_dataset.label)

training_data,training_targets=shrink_data(labeled_idxs,targets=training_dataset.label,data=training_dataset.data)


valid_data,valid_targets=shrink_data(unlabeled_idxs,targets=training_dataset.label,data=training_dataset.data)

testing_data,testing_targets=shrink_data(unlabeled_idxs,targets=training_dataset.label,data=training_dataset.data)

class Dateset(Dataset):
    def __init__(self,data,label,Training=False):
        self.len=data.shape[0]
        self.data=data
        self.label = label
        self.Training=Training



    def __getitem__(self, index):
        if self.Training:
            sample1 = self.data[index]
            sample2, label = self.get_random_sample_from_same_class(self.label[index], index)
            assert label == self.label[index]
            return (sample1,sample2),self.label[index]
        else:
            return self.data[index],self.label[index]

    def __len__(self):
        return self.len


    def get_random_sample_from_same_class(self, class_label,idx):
        class_indices = [i for i, label in enumerate(self.label) if label == class_label]

        random_index = idx
        while random_index == idx:
            random_index = np.random.choice(class_indices)

        return self.data[random_index], self.label[random_index]

if __name__ == '__main__':
    dataset=Dateset(training_data,training_targets,Training=True)
    train_loader=DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2)

    for batch_idx, ((x_l, x_l2), y_l) in enumerate(train_loader):
        print(x_l2.size())

