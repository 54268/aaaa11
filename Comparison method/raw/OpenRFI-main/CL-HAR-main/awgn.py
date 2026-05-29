

import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import torch
import torch.nn.functional as F

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

def awgn(signal,  L=1):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'noise' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        signal : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for waveform simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+noise)
    """
    SNRdB = np.random.uniform(0, 20)
    gamma = 10 ** (SNRdB / 10)  
    if signal.ndim == 1:  
        P = L * sum(abs(signal) ** 2) / len(signal)  
    else:  
        P = L * sum(sum(abs(signal) ** 2)) / len(signal)  
    N0 = P / gamma  
    if isrealobj(signal):  
        noise = sqrt(N0 / 2) * standard_normal(signal.shape)  
    else:
        noise = sqrt(N0 / 2) * (standard_normal(signal.shape) + 1j * standard_normal(signal.shape))
    
    r = signal + noise  

    return r

signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
received_signal = awgn(signal)

signal_2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

received_signal_2d = awgn(signal_2d)



result_list = []
tensor = torch.rand(32,16, 128)




for i in range(tensor.size()[0]):
    
    numpy_array = np.array(tensor[i])
    tensors=awgn(numpy_array)
    tensor1=torch.tensor(tensors)
    result_list.append(torch.squeeze(tensor1,dim=0))


sample2 = torch.stack(result_list, dim=0)
assert tensor.size() == sample2.size()
print(sample2.size())


def awgns(signal):
    device='cuda'

    
    original_image_shape = (signal.size()[0], 1, 1024, 2)
    
    restored_images = signal.view(*original_image_shape)
    restored_images = restored_images.squeeze(dim=1)

    result_list = []
    for i in range(restored_images.size()[0]):
        
        numpy_array = np.array(restored_images[i])
        signals = awgn(numpy_array)
        signals = torch.tensor(signals)
        result_list.append(torch.squeeze(signals, dim=0))

    
    signal_n = torch.stack(result_list, dim=0)


    height = 2
    transformer_embedding_size = 64

    patcher = PatchExtractor(width_size=height,
                             height_size=transformer_embedding_size,
                             width_stride_size=height,
                             height_stride_size=transformer_embedding_size).to(device)
    signal_n = signal_n.unsqueeze(dim=1)
    signal_n = patcher(signal_n)

    assert signal_n.size() == signal.size()
    return signal_n