import time
import os.path as osp
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer, get_cross_shot_dataloader, get_class_dataloader
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        # buid repo for each class
        if args.use_repo > 0:
            # copy the parameters from the basic encoder to the repo-encoder
            pretrained_dict = torch.load(args.repo_init, map_location='cpu')['params']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}                  
            pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items() if k[8:] in self.model.p_encoder.state_dict()}            
            self.model.p_encoder.load_state_dict(pretrained_dict) # also copy the running statistics
            for param_repo in self.model.p_encoder.parameters():
                param_repo.requires_grad = False  # not update by gradient
                
            # get embeddings for all training classes
            if osp.exists(osp.join(*args.repo_init.split('/')[:-1], 'RepoInit-{}.dat'.format(args.backbone_class_t))):
                RepoInit = torch.load(osp.join(*args.repo_init.split('/')[:-1], 'RepoInit-{}.dat'.format(args.backbone_class_t)))
                for c_label in RepoInit:
                    self.model.repo[c_label] = RepoInit[c_label]['emb']
                    self.model.repo_mean[c_label] = RepoInit[c_label]['center']
            else:
                self.class_loader = get_class_dataloader(args)
                self.model.eval()
                RepoInit = {}
                for batch in tqdm(self.class_loader, desc='Init P-Repo', ncols=50):
                    if torch.cuda.is_available():
                        c_data, c_label = batch[1].cuda(), batch[-1].cuda()
                    else:
                        c_data, c_label = batch[1], batch[-1]
                    unique_c_label = torch.unique(c_label)
                    assert(unique_c_label.shape[0] == 1)
                    c_label = unique_c_label.item()
                    # split the data in to shots and add them to the corresponding queue
                    with torch.no_grad():
                        inst_emb = []
                        for j in range(int(np.ceil(c_data.shape[0] / 128))):
                            inst_emb.append(self.model.p_encoder(c_data[j*128:min((j+1)*128, c_data.shape[0]), :]))
                        inst_emb = torch.cat(inst_emb)
                    self.model.repo[c_label] = inst_emb.cpu()    
                    self.model.repo_mean[c_label] = torch.mean(inst_emb, 0)
                    
                    RepoInit[c_label] = {}
                    RepoInit[c_label]['emb'] = inst_emb.cpu()    
                    RepoInit[c_label]['center'] = torch.mean(inst_emb, 0)
            
                torch.save(RepoInit, osp.join(*args.repo_init.split('/')[:-1], 'RepoInit-{}.dat'.format(args.backbone_class_t)))        
                self.model.train  
                
    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(
                    args.query
                )
        label = label.type(torch.LongTensor)
        label_aux = torch.arange(args.way, dtype=torch.int16).repeat(args.shot + args.query)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
        return label, label_aux
    
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
            self.model.p_encoder.eval()
        
        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
                self.model.p_encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            taT = Averager()
            
            start_tm = time.time()
            self.model.zero_grad()
            train_acc_T, train_acc_S, train_acc = [],[],[]
            for batch in self.train_loader:
                self.train_step += 1

                data, data_aug, gt_label = batch
                if torch.cuda.is_available():
                    data, data_aug, gt_label = data.cuda(), data_aug.cuda(), gt_label.long().cuda()
                    
                gt_label = gt_label[:args.way] # get the ground-truth label of the current episode
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                if (args.use_repo > 0) and (epoch >= args.use_repo):
                    if args.model_class in ['FEAT']:
                        logits, reg_logits, logits_s, logits_gt = self.para_model(data, data_aug, gt_label, use_repo = True)
                        loss = args.balance * F.cross_entropy(logits, label) + args.balance_feat * F.cross_entropy(reg_logits, label_aux)
                    else:
                        logits, logits_s, logits_gt = self.para_model(data, data_aug, gt_label, use_repo = True)                            
                        loss = args.balance * F.cross_entropy(logits, label)
                    loss_gt = F.kl_div(F.log_softmax(logits_s / args.kd_temperature, dim=-1), F.softmax(logits_gt / args.kd_temperature, dim=-1)) * ((args.kd_temperature) ** 2) # reduction = 'batchmean'
                    total_loss = (1-args.balance) * loss_gt + loss
                    tl2.add(loss.item())
                    acc_T = count_acc(logits_gt, label)
                    train_acc_T.append(acc_T)
                    train_acc_S.append(count_acc(logits_s, label))
                    taT.add(acc_T)
                else:
                    logits = self.para_model(data)
                    total_loss = F.cross_entropy(logits, label)
                    loss = 0
                    tl2.add(loss)
                
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)
                
                tl1.add(total_loss.item())
                ta.add(acc)
                
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)                    
                self.model.zero_grad()
                    
                train_acc.append(acc)
                if (args.use_repo > 0) and (epoch >= args.use_repo):
                    self.try_logging(tl1, tl2, ta, taT)               
                else:                
                    self.try_logging(tl1, tl2, ta)               

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            if (args.use_repo > 0) and (epoch >= args.use_repo):
                print('LOG: Epoch-{}: Train Acc-{}, Train Acc-S-{}, Train Acc-T-{}'.format(epoch, np.mean(np.stack(train_acc)).item(), np.mean(np.stack(train_acc_S)).item(), np.mean(np.stack(train_acc_T)).item()))                
            else:
                print('LOG: Epoch-{}: Train Acc-{}'.format(epoch, np.mean(np.stack(train_acc)).item()))
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
        args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        for i, batch in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
            with torch.no_grad():
                logits = self.model(data)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
            self.model.p_encoder.eval()

        args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query
        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
        args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query        
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        for i, batch in tqdm(enumerate(self.test_loader, 1)):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
                
            with torch.no_grad():
                logits = self.model(data)
                    
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
    
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
    
        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))
        
        args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query
        return vl, va, vap

    def evaluate_test_cross_shot(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()     
        num_shots = [1, 5, 10, 20, 30, 50]
        # num_shots = [1, 5]
        record = np.zeros((10000, len(num_shots))) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        for s_index, shot in enumerate(num_shots):
            test_loader = get_cross_shot_dataloader(args, shot)
            args.eval_shot = shot
            args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
            args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query        
            for i, batch in tqdm(enumerate(test_loader, 1)):
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data = batch[0]
    
                with torch.no_grad():
                    logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, s_index] = acc
            assert(i == record.shape[0])
            
            va, vap = compute_confidence_interval(record[:,s_index])
            print('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))
            args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query

        with open(osp.join(self.args.save_path, '{}+{}-CrossShot'.format(va, vap)), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))                
            for s_index, shot in enumerate(num_shots):
                va, vap = compute_confidence_interval(record[:,s_index])
                f.write('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))
                
    
    def final_record(self):
        # save the best performance in a txt file

        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))            