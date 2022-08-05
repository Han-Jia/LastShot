import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet(dropblock_size=args.dropblock_size)
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res50':
            hdim = 2048
            # from model.networks.resnet50 import res50
            # self.encoder = res50()
            from model.networks.resnet_torch import resnet50
            self.encoder = resnet50()            
        elif args.backbone_class == 'Res101':
            hdim = 640
            from model.networks.res12 import ResNetDeeper
            self.encoder = ResNetDeeper(dropblock_size=args.dropblock_size)
            # hdim = 2048
            # from model.networks.res18 import resnet101
            # self.encoder = resnet101()                   
        elif args.backbone_class == 'Res152':
            hdim = 2048
            from model.networks.res18 import resnet152
            self.encoder = resnet152()   
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)

    def forward(self, data, is_emb = False):
        out = self.encoder(data)
        # import pdb
        # pdb.set_trace()
        if not is_emb:
            out = self.fc(out)
        return out
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim
    
    def forward_proto_mcrop(self, data_shot, data_query, way = None):
        # for multiple crop
        # data_shot: (way*shot) * crop * 3 x 84 x 84
        # data_query (way*query) * crop * 3 x 84 x 84
        size_data = data_shot.shape[-3:]
        num_crop = data_shot.shape[1]
        num_query = data_query.shape[0]
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot.view(-1, *size_data))
        size_dim = proto.shape[-1]
        proto = proto.reshape(-1, way, num_crop, size_dim).mean(dim=0)
        # average the center of multiple crops
        proto = proto.mean(dim=1)
        query = self.encoder(data_query.view(-1, *size_data))
        query = query.reshape(-1, size_dim) # (query * crop) * dim
        ## ensemble over the queries
        # distance
        logits_dist = euclidean_metric(query, proto).view(num_query, num_crop, -1)
        logits_dist = logits_dist.mean(dim=1)
        # similarity
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t()).view(num_query, num_crop, -1)
        logits_sim = logits_sim.mean(dim=1)        
        return logits_dist, logits_sim    