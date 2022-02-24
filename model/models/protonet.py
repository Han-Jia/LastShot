import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, instance_embs, instance_embs_aug, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        if self.args.aug_support:
            support = instance_embs_aug[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        else:
            support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            
        if self.args.aug_query:
            query   = instance_embs_aug[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        else:
            query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if True: # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits
