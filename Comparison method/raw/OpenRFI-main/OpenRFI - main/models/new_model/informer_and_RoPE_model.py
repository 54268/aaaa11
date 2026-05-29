import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .attn_with_RoPE import ProbAttention, AttentionLayer
from .encoder_with_RoPE import EncoderLayer
from thop import profile

class PatchExtractor(nn.Module):
    
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(d_model, max_seq_len).to(device)

    def get_positional_encoding(self, d_model, max_seq_len):
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        positional_encoding = torch.zeros((max_seq_len, d_model), dtype=torch.float32)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :] 


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):                    
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)                     
        x = self.activation(x)
        x = self.maxPool(x)                  
        x = x.transpose(1,2)                 
        return x                             

class informer(nn.Module):  
    def __init__(self,  d_model, num_heads, max_seq_len,encoderlayer,conv_layers):
        super(informer, self).__init__()
        self.attn_layer1 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
                                                    d_model=d_model, n_heads=num_heads, mix=False),
                                                    d_model=d_model,
                                                    d_ff=512,
                                                    dropout=0.0,
                                                    activation='gelu')  
        self.conv_layer1 = conv_layers(d_model)

        self.attn_layer2 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
                                                    d_model=d_model, n_heads=num_heads, mix=False),
                                                    d_model=d_model,
                                                    d_ff=512,
                                                    dropout=0.0,
                                                    activation='gelu')
        self.conv_layer2 = conv_layers(d_model)

        self.attn_layer3 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.1, output_attention=False),
                                                    d_model=d_model, n_heads=num_heads, mix=False),
                                                    d_model=d_model,
                                                    d_ff=512,
                                                    dropout=0.0,
                                                    activation='gelu')
        self.conv_layer3 = conv_layers(d_model)


        self.norm = nn.LayerNorm(d_model)

        self.embedding = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, x, attn_mask=None):
        

        x = self.embedding(x)
        

        attns = []

        x, attn = self.attn_layer1(x, attn_mask=attn_mask)
        x = self.conv_layer1(x)  
        attns.append(attn)

        x, attn = self.attn_layer2(x, attn_mask=attn_mask)
        x = self.conv_layer2(x)  
        attns.append(attn)


        x, attn = self.attn_layer2(x, attn_mask=attn_mask)  
        attns.append(attn)

        return x,attns

class TransformerModel(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self,  device='cuda'):
        super(TransformerModel, self).__init__()

        
        self.patcher = PatchExtractor(width_size=2,
                                      height_size=64,
                                      width_stride_size=2,
                                      height_stride_size=128).to(device)

        
        self.model = informer(d_model=128, num_heads=4, max_seq_len=1024//128+1,encoderlayer=EncoderLayer,conv_layers=ConvLayer)
        
        self.pool = nn.AdaptiveAvgPool1d(1).to(device)
        self.fc1 = torch.nn.Linear(128,32)  

    def classification(self, inputs):
        
        x = inputs.unsqueeze(dim=1)  
        patch = self.patcher(x)  
        
        enc_out,_ = self.model(patch)
        out = enc_out.permute(0, 2, 1)  
        pooling = self.pool(out)        
        flatten = pooling.squeeze(-1)   
        y = self.fc1(flatten)           

        return y

    def forward(self, inputs, mask=None):
        
        dec_out = self.classification(inputs)
        return dec_out  

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    
    batch_size = 1
    num_seq = 32           
    d_model = 64
    num_heads = 4
    max_seq_len = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = informer(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,encoderlayer=EncoderLayer,conv_layers=ConvLayer)
    model = model.to(device)

    
    print(get_parameter_number(model))


    
    input_data = torch.randn(batch_size, num_seq, d_model)
    input_data = input_data.to(device)
    

    
    flops, params = profile(model, inputs=(input_data,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    
    output,_ = model(input_data)

    


