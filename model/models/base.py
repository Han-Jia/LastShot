import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def split_instances(num_tasks, num_shot, num_query, num_way, num_class=None):
    num_class = num_way if (num_class is None or num_class < num_way) else num_class

    permuted_ids = torch.zeros(num_tasks, num_shot+num_query, num_way).long()
    cls_list = []
    for i in range(num_tasks):
        # select class indices
        clsmap = torch.randperm(num_class)[:num_way]
        cls_list.append(clsmap)
        # ger permuted indices
        for j, clsid in enumerate(clsmap):
            permuted_ids[i, :, j].copy_(
                torch.randperm((num_shot + num_query)) * num_class + clsid
            )

    cls_list = torch.stack(cls_list)
    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()
        cls_list = cls_list.cuda()

    support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
    return support_idx, query_idx, cls_list

class ResidualScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Identity()
        self.projection = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.projection(x) + self.shortcut(x)
    
class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            hdim = 64
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet(dropblock_size=args.dropblock_size) 
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)
        else:
            raise ValueError('')
        
        if args.backbone_class_t == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.p_encoder = ConvNet()
        elif args.backbone_class_t == 'Res12':
            from model.networks.res12 import ResNet
            self.p_encoder = ResNet(dropblock_size=args.dropblock_size)
        elif args.backbone_class_t == 'Res18':
            from model.networks.res18 import ResNet
            self.p_encoder = ResNet()  
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.p_encoder = Wide_ResNet(28, 10, 0.5)
        elif args.backbone_class_t == 'Res101':
            from model.networks.resnet_torch import resnet101
            self.p_encoder = resnet101() 
        elif args.backbone_class_t == 'Res152':
            from model.networks.resnet_torch import resnet152
            self.p_encoder = resnet152()             
        else:
            raise ValueError('')        

        # optimize a shared scale and bias
        self.scale = ResidualScale(args.way, args.way)
        self.hdim = hdim
        self.repo = {}
        self.repo_mean = {}
        
    def split(self):
        args = self.args
        if self.training:
            return  split_instances(1, args.shot, args.query, args.way)
        else:
            return  split_instances(1, args.eval_shot, args.eval_query, args.eval_way)

    def forward(self, x, x_aug=None, gt_label=None, get_feature=False, use_repo = False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            if x_aug is not None:
                x_aug = x_aug.squeeze(0)
                # instance_embs_aug = self.encoder(x_aug)
            else:
                x_aug = x
                # instance_embs_aug = instance_embs
                
            # split support query set for few-shot data
            support_idx, query_idx, cls_list = self.split()
            if (self.training) and self.args.model_class in ['FEAT']:
                logits, logits_reg = self._forward(instance_embs, instance_embs, support_idx, query_idx)    # No Aug on vanilla meta-methods
                logits_s = logits
            else:
                logits = self._forward(instance_embs, instance_embs, support_idx, query_idx)
                logits_s = logits
                
            if (self.training) and (use_repo== True):
                # Set the teacher to Train Mode
                self.p_encoder.eval()
                with torch.no_grad():
                    instance_embs_p = self.p_encoder(x_aug) # instance_embs_aug
                    
                support_set = [torch.cat([self.repo[cls_label.item()][torch.randperm(self.repo[cls_label.item()].shape[0])[:50], :] for cls_label in c_list]) for c_list in gt_label[cls_list]]
                support_label = [torch.cat([cls_index * torch.ones(50, ) for cls_index, cls_label in enumerate(c_list)]) for c_list in gt_label[cls_list]]
    
                if self.args.kd_type == 'LR':
                    logits_gt = self._kd_forward_lr(instance_embs_p, query_idx, support_set, support_label)
                elif self.args.kd_type == 'SVM':
                    logits_gt = self._kd_forward_svm(instance_embs_p, query_idx, support_set, support_label)                    
                else:
                    task_mean = [torch.cat([self.repo_mean[cls_label.item()].view(1, -1) for cls_label in c_list]) for c_list in gt_label[cls_list]]
                    logits_gt = self._kd_forward_ncm(instance_embs_p, query_idx, task_mean)
                
                logits_s = self.scale(logits_s)
                
                if self.args.model_class in ['FEAT']:
                    return logits, logits_reg, logits_s, logits_gt
                else:
                    return logits, logits_s, logits_gt
            else:
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')
    
    def _kd_forward_ncm(self, instance_embs, query_idx, support_center):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        
        # apply NCM for all tasks
        num_tasks = len(support_center)
        support_center = torch.stack(support_center)
        # normalization
        support_center = nn.functional.normalize(support_center, dim=-1)
        query = query.view(num_tasks, -1, emb_dim)
        logits = torch.bmm(query, support_center.permute([0,2,1]))
        num_class = logits.shape[-1]
        logits = logits.view(-1, num_class)
        return logits     
    
    def _kd_forward_lr(self, instance_embs, query_idx, support_set, support_label):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        
        # train LR for each task
        num_task = len(support_set)        
        logits = []
        for task_inds in range(num_task):
            current_support = support_set[task_inds].view(-1, emb_dim).detach().cpu().numpy()
            current_query = query[task_inds, :].view(-1, emb_dim)
            clf = LogisticRegression(random_state=0, dual=False, max_iter=10000, C=self.args.kd_c, multi_class='multinomial')
            clf.fit(current_support, support_label[task_inds])
            logits.append(torch.Tensor(clf.decision_function(current_query.detach().cpu().numpy())))

        logits = torch.cat(logits, 0)
        if torch.cuda.is_available():
            logits = logits.cuda()
        return logits 
    
    
    def _kd_forward_svm(self, instance_embs, query_idx, support_set, support_label):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        
        # train SVM for each task
        num_task = len(support_set)        
        logits = []
        for task_inds in range(num_task):
            current_support = support_set[task_inds].view(-1, emb_dim).detach().cpu().numpy()
            current_query = query[task_inds, :].view(-1, emb_dim)            
            clf = LinearSVC(random_state=0, dual=False, max_iter=10000, C=self.args.kd_c, multi_class='crammer_singer')
            clf.fit(current_support, support_label[task_inds])
            logits.append(torch.Tensor(clf.decision_function(current_query.detach().cpu().numpy())))

        logits = torch.cat(logits, 0)
        if torch.cuda.is_available():
            logits = logits.cuda()
        return logits 
    
