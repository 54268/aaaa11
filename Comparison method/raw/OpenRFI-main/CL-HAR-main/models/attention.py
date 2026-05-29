import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
import math
from models.new_model.attn_with_RoPE import ProbAttention, AttentionLayer
from models.new_model.encoder_with_RoPE import EncoderLayer

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        self.attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', self.attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

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

class roinformer(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len,d_ff,dropout,encoderlayer, conv_layers):
        super().__init__()
        self.attn_layer1 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                                                                 d_model=d_model,
                                                                 n_heads=num_heads,
                                                                 mix=False),
                                        d_model=d_model,
                                        d_ff=d_ff,
                                        dropout=dropout,
                                        activation='gelu')
        self.conv_layer1 = conv_layers(d_model)
        self.attn_layer2 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                                                                 d_model=d_model,
                                                                 n_heads=num_heads,
                                                                 mix=False),
                                        d_model=d_model,
                                        d_ff=d_ff,
                                        dropout=dropout,
                                        activation='gelu')
        self.conv_layer2 = conv_layers(d_model)
        self.attn_layer3 = encoderlayer(attention=AttentionLayer(ProbAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                                                                 d_model=d_model,
                                                                 n_heads=num_heads,
                                                                 mix=False),
                                        d_model=d_model,
                                        d_ff=d_ff,
                                        dropout=dropout,
                                        activation='gelu')
        self.conv_layer3 = conv_layers(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.embedding = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout,max_seq_len)


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
        return x

class Seq_Transformer(nn.Module):
    def __init__(self, n_channel, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()
        self.classifier = nn.Linear(dim, n_classes)


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        x = self.position(x)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b) 
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class Seq_RoInformer(nn.Module):
    def __init__(self, n_channel, len_sw, n_classes, dim=128, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(n_channel, dim)
        self.position = PositionalEncoding(d_model=dim, max_len=len_sw)
        self.roinformer = roinformer(d_model=dim, num_heads=heads, max_seq_len=len_sw,dropout=dropout,d_ff=d_ff,
                                     encoderlayer=EncoderLayer,conv_layers=ConvLayer)
        self.classifier = nn.Linear(dim, n_classes)
        self.pool = nn.AdaptiveAvgPool1d(1).to('cuda')  


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)

        x = self.roinformer(x)
        x = x.permute(0, 2, 1)
        c_t = self.pool(x).squeeze(-1)
        return c_t
class _Seq_Transformer(nn.Module): 
    def __init__(self, patch_size, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super().__init__()
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

if __name__ == "__main__":
    
    batch_size = 64
    num_seq = 128
    d_model = 128
    num_heads = 4
    max_seq_len = 100
    n_channel = 9
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq_RoInformer(n_channel=n_channel, len_sw=100, n_classes=32, dim=d_model, heads=num_heads, d_ff=512, dropout=0.1)
    model = model.to(device)


    input_data = torch.randn(batch_size, num_seq, n_channel)
    input_data = input_data.to(device)
    print(input_data.shape)

    output = model(input_data)
    print(output.shape)