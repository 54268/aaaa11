from dataset.data_preprocess.data_prep_RF.rff_dataset_amc_load import Datasets
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from dataset.data_preprocess.base_loader import base_loader
from itertools import cycle
from dataset.data_preprocess.augmentations import gen_aug


def datatransformer (training_dataset):
    data_list=[]
    label_list=[]
    domain_list=[]

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

multiple_files_training_dataset_dir = './dataset/data/' + dataset_dir_name + '/training'
multiple_files_valid_dataset_dir = './dataset/data/' + dataset_dir_name + '/valid'
multiple_files_test_dataset_dir = './dataset/data/' + dataset_dir_name + '/valid'  







h5_file_data_set_filename = 'noisy_datas'
h5_file_label_filename = 'labels'
h5_file_snr_filename = 'snrs'
sample_size_dim = 0


training_dataset = Datasets().get_dataset(filename=multiple_files_training_dataset_dir,h5_file_data_set_filename='original_datas',h5_file_label_filename='labels', h5_file_snr_filename='snrs',width=1024,code_mode='training')



data_list,label_list,domain_list=datatransformer(training_dataset) 


def get_labeled_indexes( labeled_classes, labeled_ratio,targets):
    
    
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

def shrink_datas( idxs,targets,data,domain):  
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
"""
源域大量标签数据（24类），目标域少量已知类标签数据（24类）+大量无标签已知类数据（24类）+大量无标签未知类数据（8类）
"""



classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

labeled_num=10 
labeled_classes = range(labeled_num)


labeled_idxs, unlabeled_idxs = get_labeled_indexes(labeled_classes, labeled_ratio=1.01,targets=label_list)


training_data,training_targets,training_domain =shrink_datas(labeled_idxs,targets=label_list,data=data_list,domain=domain_list)


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

def prep_rf_random(SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    split_ratio = 0.2
    batch_size=256

    x_win_all, y_win_all, d_win_all=training_data,training_targets,training_domain
    

    height = 2
    transformer_embedding_size = 64  
    width = 1024

    d_model = height * transformer_embedding_size  
    seq_len = width // transformer_embedding_size
    max_seq_len = width // transformer_embedding_size + 1

    patcher = PatchExtractor(width_size=height,
                             height_size=transformer_embedding_size,
                             width_stride_size=height,
                             height_stride_size=transformer_embedding_size).to(device)

    x_win_all = torch.from_numpy(x_win_all)  
    x_win_all = x_win_all.unsqueeze(dim=1)
    x_win_all = patcher(x_win_all)  
    x_win_all = x_win_all.numpy()

    x_win_all = np.transpose(x_win_all.reshape((-1, 1, seq_len, d_model)),
                             (0, 2, 1, 3))  

    
    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=split_ratio)

    
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

    
    train_loader_r = DataLoader(train_set_r, batch_size=batch_size,  drop_last=True, sampler=sampler)
    
    


    val_set_r = data_loader_rf(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=batch_size, shuffle=False)
    test_set_r = data_loader_rf(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=batch_size, shuffle=False)

    return train_loader_r, val_loader_r, test_loader_r


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


class new_dateset(Dataset):
    def __init__(self,data,label,Training=False):
        self.len=len(data)
        self.data=data
        self.label = label
        self.Training= Training
        
        

    def __getitem__(self, index): 
        if self.Training:
            
            
            sample1 = self.data[index]
            sample2 = self.data[index]
            return (sample1,sample2),self.label[index] 
        else:
            return self.data[index],self.label[index] 

    def __len__(self): 
        return self.len


def rebuild_dataset(labeled_num,labeled_ratio):
    train_loader, val_loader, test_loader =prep_rf_random()

    data = []
    label = []
    for idx, (sample, target, domain) in enumerate(train_loader):
        data.append(sample)
        label.append(target)
    for idx, (sample, target, domain) in enumerate(val_loader):
        data.append(sample)
        label.append(target)
    for idx, (sample, target, domain) in enumerate(test_loader):
        data.append(sample)
        label.append(target)

    data = torch.cat(data, dim=0)
    labels = torch.cat(label, dim=0)

    """
    训练集数据+验证集数据 所有类中的一半作为已知类，已知类中10%的样本已知标签  半监督学习    2/4 0.1
    """
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    labeled_classes = range(labeled_num)
    
    labeled_idxs, unlabeled_idxs = get_labeled_index(labeled_classes, labeled_ratio=labeled_ratio, targets=labels)

    train_labeled_data, train_labelded_targets = shrink_data(labeled_idxs, targets=labels,data=data)
    train_unlabeled_data, train_unlabel_targets = shrink_data(unlabeled_idxs, targets=labels, data=data)
    test_data, test_targets = shrink_data(unlabeled_idxs, targets=labels,data=data)

    print('labeled_num: ',len(train_labeled_data))
    print('unlabeled_num: ',len(train_unlabeled_data)) 


    
    train_label_data = []
    train_unlabel_data = []
    testing_data = []
    for i in range(train_labeled_data.size()[0]):
        train_label_data.append(train_labeled_data[i,:,:].squeeze(dim=0))
    for i in range(train_unlabeled_data.size()[0]):
        train_unlabel_data.append(train_unlabeled_data[i,:,:].squeeze(dim=0))
    for i in range(test_data.size()[0]):
        testing_data.append(test_data[i,:,:].squeeze(dim=0))

    train_labeled_dataset = new_dateset(train_labeled_data, train_labelded_targets,Training=True)
    train_unlabeled_dataset = new_dateset(train_unlabel_data, train_unlabel_targets,Training=True)
    test_dataset = new_dateset(test_data, test_targets,Training=False)


    return train_labeled_dataset,train_unlabeled_dataset,test_dataset
    
    
    
       


if __name__ == '__main__':
    train_label_set, train_unlabel_set, test_set = rebuild_dataset(labeled_num=5,labeled_ratio=0.1)

    batch_size = 128

    labeled_len = len(train_label_set)
    print("labeled_len",labeled_len)
    unlabeled_len = len(train_unlabel_set)
    print("unlabeled_len", unlabeled_len)
    labeled_batch_size = int(batch_size * labeled_len / (labeled_len + unlabeled_len))
    print("labeled_batch_size", labeled_batch_size)

    
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=8, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)

    for batch_idx, (x, label) in enumerate(test_loader):
        print("x.size()", x.size())
        break

    for batch_idx, ((x_l, x_l2), y_l) in enumerate(train_label_loader):
        x_l = gen_aug(x_l, "awgn")
        x_l2 = gen_aug(x_l, "perm_jit_RF")
        print("x_l.size()", x_l.size())
        print("x_l2.size()", x_l2.size())

        unlabel_loader_iter = cycle(train_unlabel_loader)
        ((x_u, x_u2), y_u) = next(unlabel_loader_iter)
        x_u = gen_aug(x_u,"awgn")
        x_u2 = gen_aug(x_u2,"perm_jit_RF")
        print("x_u.size()", x_u.size())
        break
