from models.new_model.backbones import *
import torch.nn as nn
import copy
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))

        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x

class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2

def setup_linclf(bb_dim):
    n_class = 10
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier

def setup_model_optm(classifier=True):
    backbone = 'RoInformer'
    n_feature = 128
    len_sw = 17
    n_class = 10
    framework = 'simclr'
    p = 128
    lr=3e-3
    
    if backbone == 'Transformer':
        backbone = Transformer(n_channels=n_feature, len_sw=len_sw, n_classes=n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif backbone == 'RoInformer':
        backbone = RoInformer(n_channels=n_feature, len_sw=len_sw, n_classes=n_class, dim=128,heads=4,d_ff=512, dropout=0.1, backbone=False)
    else:
        NotImplementedError

    
    if framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=p)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizers = [optimizer]

    else:
        NotImplementedError

    model = model.to(DEVICE)

    
    if classifier:
        bb_dim = backbone.out_dim
        classifier = setup_linclf(bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers

def lock_backbone(model):
    framework = 'simclr'
    for name, param in model.named_parameters():
        param.requires_grad = False

    if framework =='simclr' :
        trained_backbone = model.encoder
    else:
        NotImplementedError

    return trained_backbone

class Roinformer(nn.Module):
    def __init__(self, backbone, dim=32):
        super(Roinformer, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
    def forward(self, x):
        _, z = self.encoder(x)

        if len(z.shape) == 3:
            z = z.reshape(z.shape[0], -1)
        z = self.projector(z)

        return z

def get_trained_backbone():
    n_feature = 128
    len_sw = 17
    n_class = 10

    backbone = RoInformer(n_channels=n_feature, len_sw=len_sw, n_classes=n_class, dim=128, heads=4, d_ff=512,
                          dropout=0.1, backbone=False)
    best_pretrain_model = SimCLR(backbone=backbone, dim=128)

    checkpoint = torch.load(
        '..\pretrained\class_n_10.pt')  
    best_pretrain_model.load_state_dict(checkpoint['model_state_dict'])  
    trained_backbone = lock_backbone(best_pretrain_model)  
    return trained_backbone



if __name__ == '__main__':


    roinformer=Roinformer(get_trained_backbone())

    tensor = torch.rand(100, 16, 128)

    out=roinformer(tensor)

    print(out.size())


    for name, param in roinformer.encoder.named_parameters():
        if "classifier.weight" or "classifier.bias" == name:
            print(name)
    """
    for name, param in roinformer.encoder.named_parameters():
        if 'classifier' not in name :
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(name)
    """
