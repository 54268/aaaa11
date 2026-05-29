import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.roinformer import Roinformer
from models.roinformer import get_trained_backbone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature=0.1, use_cosine_similarity=True,device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        
        
        
        return v

    def _cosine_simililarity(self, x, y):
        
        
        
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)



class Model(nn.Module):
    def __init__(self, arch='roinformer', proto_num=50, latent_dim=32, tau=0.1, init=False, device='cuda'):
        super(Model, self).__init__()
        self.arch = arch
        self.proto_num = proto_num
        self.latent_dim = latent_dim
        self.tau = tau
        self.encoder = Roinformer(get_trained_backbone())
        
        self.prototypes = nn.Parameter(torch.randn(self.proto_num, self.latent_dim).to(device), requires_grad=True)
        
        self.proto_ind = torch.ones(len(self.prototypes), dtype=torch.bool)
        
        self.group_mask = torch.eye(len(self.prototypes)).to(device)
        
        self.proto_graph = torch.eye(self.proto_num).to(device)
        self.proto_mask = torch.eye(self.proto_num).to(device)

        if init:
            self.prototypes.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def find_pairs(self, feature, y_l):
        
        feat_detach = feature.detach().flatten(start_dim=1)
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t()) 
        labeled_len = len(y_l)
        pos_pairs = []
        target_np = y_l.cpu().numpy()
        
        for i in range(labeled_len):
            target_i = target_np[i]
            
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)  
                while selec_idx == i:                  
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
          
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)  
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
        return pos_pairs     


    def find_positive_samples(self, batch_size,y_l):

        pos_positive_samples=[i for i in range(0, batch_size)]

        labeled_len = len(y_l)  
        target_np = y_l.cpu().numpy()  

        for i in range(labeled_len):  
            target_i = target_np[i]  
            
            idxs = np.where(target_np == target_i)[0]  
            if len(idxs) == 1:  
                pos_positive_samples[i]=idxs[0]
            else:  
                selec_idx = np.random.choice(idxs, 1)  
                while selec_idx == i:  
                    selec_idx = np.random.choice(idxs, 1)
                pos_positive_samples[i]=int(selec_idx)

        return pos_positive_samples  


    def loss(self, x_l, x_l2, y_l, x_u, x_u2, labeled_class, conf, weight):
        labeled_len = len(y_l)
        
        c = self.prototypes
        c = F.normalize(c, dim=1)

        x = torch.cat([x_l, x_u], 0)
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        
        p = F.softmax(torch.mm(z, c.t()) / self.tau, dim=1)  
        
        
        q = torch.mm(p, self.group_mask.t()) 

        
        x2 = torch.cat([x_l2, x_u2], 0)
        z2 = self.encoder(x2)
        z2 = F.normalize(z2, dim=1)
        p2 = F.softmax(torch.mm(z2, c.t()) / self.tau, dim=1)
        q2 = torch.mm(p2, self.group_mask.t()) 

        
        pair_ind = self.find_pairs(z, y_l) 
        p2_pair = p2[pair_ind, :]         
        
        
        proto_sim1 = torch.mean(torch.sum(- p2_pair * torch.log(p + 1e-8), dim=1))
        proto_sim2 = torch.mean(torch.sum(- p * torch.log(p2_pair + 1e-8), dim=1))
        
        
        proto_sim = 0.5 * proto_sim1 + 0.5 * proto_sim2


        
        q2_pair = q2[pair_ind, :]
        group_sim_1 = torch.mean(torch.sum(- q2_pair * torch.log(q + 1e-8), dim=1))
        
        
        group_sim_2 = torch.mean(torch.sum(- q * torch.log(q2 + 1e-8), dim=1))
        group_sim = 0.5 * group_sim_1 + 0.5 * group_sim_2              
        

        
        y_l_onehot = F.one_hot(y_l, len(self.group_mask)) 
        cls_loss = torch.sum(- y_l_onehot * torch.log(q[:labeled_len] + 1e-8)) / len(y_l) 

        
        
        p_prior = F.normalize(F.normalize(self.group_mask, p=1, dim=1).sum(0), p=1, dim=0)
        p_proto = p.mean(0)
        ent_loss = torch.sum(p_proto * torch.log(p_proto / p_prior) + 1e-7) * 10  
        

        
        
        
        bs=z.size()[0]
        positive_samples = self.find_positive_samples(bs,y_l)
        pos = z2[positive_samples, :]
        criterion = NTXentLoss(bs, temperature=0.05)
        new_loss = criterion(z, pos)

        
        
        M_mask = torch.ones_like(q[:, 0], dtype=torch.bool) 
        M_mask[:labeled_len] = False
        
        max_values, max_indices = torch.max(q, dim=1) 
        
        S_mask = max_values > conf  
        Y_mask = max_indices >= labeled_class 
        
        Mask = np.logical_and(np.logical_and(M_mask.cpu(), S_mask.cpu()), Y_mask.cpu())
        Mask = Mask.to(torch.bool)
        if Mask.any():
            
            q_masked = q[Mask]
            
            q_proto = q_masked.mean(0)  
            
            q_prior = torch.full((q_proto.shape[0],), 1 / q_proto.shape[0]).to(device)
            q_ent_loss = torch.sum(q_proto * torch.log(q_proto / q_prior) + 1e-7) * weight   
        else:
            q_ent_loss = torch.tensor(0)

        loss = {'proto': proto_sim, 'group': group_sim, 'cls': cls_loss, 'ent': ent_loss, 'new' : new_loss, 'q_ent' : q_ent_loss}

        return loss


    def pred(self, x):
        z = self.encoder(x) 
        c = self.prototypes 
        z = F.normalize(z, dim=1)
        c = F.normalize(c, dim=1)
        dist = torch.mm(z, c.t())

        p = F.softmax(dist / self.tau, dim=1)
        conf, pred_proto = torch.max(p, axis=1)
        pred_onehot = torch.mm(F.one_hot(pred_proto, len(c)).float(), self.group_mask.t())  
        pred = torch.argmax(pred_onehot, dim=1)
        
        return pred, conf, z





