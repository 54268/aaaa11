import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  
    theta = torch.pow(10000, -2 * ids / output_dim)

    
    embeddings = position * theta  

    
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  

    
    
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings




def RoPE(q, k):
    
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    
    
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  

    
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  



    
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    
    k = k * cos_pos + k2 * sin_pos

    return q, k




def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    
    
    

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logits = torch.matmul(q, k.transpose(-2, -1))  
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)  

    att_scores = F.softmax(att_logits, dim=-1)  

    if dropout is not None:
        att_scores = dropout(att_scores)

    
    return torch.matmul(att_scores, v), att_scores


if __name__ == '__main__':
    
    q = torch.randn((8, 12, 10, 32))
    k = torch.randn((8, 12, 10, 32))
    v = torch.randn((8, 12, 10, 32))

    res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)


    
    

    q = torch.ones((1, 1, 6, 16))
    k = torch.ones((1, 1, 6, 16))
    v = torch.ones((1, 1, 6, 16))

    att = torch.matmul(q, k.transpose(-2, -1))  
    q,k = RoPE(q, k)
    atts = torch.matmul(q, k.transpose(-2, -1))  
    print("att:",att)
    print("atts:",atts)
