from data_preprocess.data_prep_RF.rff_dataset_amc_load import Datasets
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader
from itertools import cycle

def datatransformer (training_dataset):
    data_list=[]
    label_list=[]
    domain_list=[]

    print(len(training_dataset.label))
    for i in range (len(training_dataset.label)):
        if training_dataset.label[i] == 2:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 6
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 5:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 1
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 10:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 2
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 12:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 5
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 15:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 9
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 17:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 3
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 25:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 4
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 26:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 0
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 27:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 8
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])

        if training_dataset.label[i] == 31:
            data_list.append(training_dataset.data[i])
            training_dataset.label[i] = 7
            label_list.append(training_dataset.label[i])
            domain_list.append(training_dataset.snr[i])



    return data_list,label_list,domain_list

device='cuda'
dataset_dir_name = '2022_12/steady_signal'

multiple_files_training_dataset_dir = '../data/' + dataset_dir_name + '/training'

multiple_files_valid_dataset_dir = '../data/' + dataset_dir_name + '/valid'
multiple_files_test_dataset_dir = '../data/' + dataset_dir_name + '/valid'  


h5_file_data_set_filename = 'noisy_datas'
h5_file_label_filename = 'labels'
h5_file_snr_filename = 'snrs'
sample_size_dim = 0


training_dataset = Datasets().get_dataset(filename=multiple_files_training_dataset_dir,h5_file_data_set_filename='original_datas',h5_file_label_filename='labels', h5_file_snr_filename='snrs',width=1024,code_mode='training')

data_list,label_list,domain_list=datatransformer(training_dataset)

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

def shrink_data( idxs,targets,data,domain):
    targets = np.array(targets)
    targets = targets[
        idxs].tolist()
    data = np.array(data)
    data = data[
        idxs, ...]
    domain = np.array(domain)
    domain = domain[
        idxs, ...]
    return data,targets,domain

def get_class_index( n_class, targets):

    class_idxs = []
    for idx, label in enumerate(targets):
        if label == n_class :
            class_idxs.append(idx)
    return class_idxs

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
labeled_num=10
labeled_classes = range(labeled_num)


labeled_idxs, unlabeled_idxs = get_labeled_index(labeled_classes, labeled_ratio=1.01,targets=label_list)

training_data,training_targets,training_domain =shrink_data(labeled_idxs,targets=label_list,data=data_list,domain=domain_list)


class PatchExtractor(torch.nn.Module):

    def __init__(self, width_size, height_size, width_stride_size, height_stride_size):

        super(PatchExtractor, self).__init__()

        self.width_patch_size = width_size
        self.height_patch_size = height_size
        self.width_stride_size = width_stride_size
        self.height_stride_size = height_stride_size


    def forward(self, images):

        patches = F.unfold(
            images,
            kernel_size=(self.height_patch_size, self.width_patch_size),
            stride=(self.height_stride_size, self.width_stride_size),
        )

        patches = patches.transpose(1, 2).contiguous()
        return patches

class data_loader_rf(base_loader):
    def __init__(self, samples, labels, domains, t):
        super(data_loader_rf, self).__init__(samples, labels, domains)
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = self.T(sample)
        return np.squeeze(np.transpose(sample, (1, 0, 2))), target, domain


dataloader_library = []

def prep_rf_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    x_win_all, y_win_all, d_win_all=training_data,training_targets,training_domain


    d_model = args.height * args.transformer_embedding_size
    seq_len = args.width // args.transformer_embedding_size
    max_seq_len = args.width // args.transformer_embedding_size + 1

    patcher = PatchExtractor(width_size=args.height,
                             height_size=args.transformer_embedding_size,
                             width_stride_size=args.height,
                             height_stride_size=args.transformer_embedding_size).to(device)

    x_win_all = torch.from_numpy(x_win_all)
    x_win_all = x_win_all.unsqueeze(dim=1)
    x_win_all = patcher(x_win_all)  
    x_win_all = x_win_all.numpy()

    x_win_all = np.transpose(x_win_all.reshape((-1, 1, seq_len, d_model)),
                             (0, 2, 1, 3))


    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)


    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=np.zeros(d_model), std=np.ones(d_model))
    ])

    train_set_r = data_loader_rf(x_win_train, y_win_train, d_win_train, transform)


    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,  drop_last=True, sampler=sampler)

    global dataloader_library

    samples, targets, domains = train_set_r.samples,train_set_r.labels,train_set_r.domains

    for i in range(10):
        idx = get_class_index(i, targets)
        sample, target, domain = shrink_data(idx, targets, samples, domains)
        train_set = data_loader_rf(sample, target, domain, transform)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        loader_iter = cycle(train_loader)
        dataloader_library.append(loader_iter)

    
    assert len(dataloader_library) == 10, "列表中的元素数量不等于10"


    val_set_r = data_loader_rf(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_rf(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_rf(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_rf_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'



if __name__ == '__main__':

    print(len(training_targets))
