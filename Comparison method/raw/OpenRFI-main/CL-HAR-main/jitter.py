import torch
import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal
import torch.nn.functional as F

def awgn_pytorch(signal: torch.Tensor, L: float = 1, device='cuda') -> torch.Tensor:


    
    SNRdB = torch.rand(signal.size(0)) * 20

    
    gamma = 10 ** (SNRdB / 10).unsqueeze(-1).unsqueeze(-1).to(device)

    
    signal_power = signal.pow(2).sum(dim=-1, keepdim=True)

    
    P = L * signal_power.mean(dim=1, keepdim=True)

    
    N0 = P / gamma

    
    noise = torch.sqrt(N0 / 2) * torch.randn_like(signal)

    
    r = signal + noise

    return r


def awgn_numpy(signal, L=1):
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


if __name__ == "__main__":
    device = 'cuda'
    
    signal = torch.randn(10, 1024, 2)  
    signal = signal.to(device)
    
    
    received_signal = awgn_pytorch(signal)

    print("Input Signal Dimension: ", signal.shape)
    print("Output Signal Dimension:", received_signal.shape)

    height=2
    transformer_embedding_size=64

    patcher = PatchExtractor(width_size=height,
                             height_size=transformer_embedding_size,
                             width_stride_size=height,
                             height_stride_size=transformer_embedding_size).to(device)

    signal = signal.unsqueeze(dim=1)
    signals = signal
    print(signal.shape)
    signal = patcher(signal)
    print(signal.shape)

    original_image_shape = (signal.size()[0], 1, 1024, 2)

    
    restored_images = signal.view(*original_image_shape)
    print(restored_images.shape)

    print(torch.all(torch.eq(restored_images,signals)).item())

    restored_images=restored_images.squeeze(dim=1)
    print(restored_images.shape)