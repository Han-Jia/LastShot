import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
# /amax/data/tiered_imagenet_raw
IMAGE_PATH1 = '/mnt/data51/tiered_imagenet_raw'
SPLIT_PATH = osp.join(ROOT_PATH, 'data/tieredimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
split_map = {'train':IMAGE_PATH1, 'val':IMAGE_PATH1, 'test':IMAGE_PATH1}

def identity(x):
    return x

class tieredImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

        self.use_im_cache = ( im_size != -1 ) # not using cache
        self.augment = augment
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path, setname)
                self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label }, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(csv_path, setname)

        self.num_class = len(set(self.label))

        image_size = 84
        self.enlarge_list = [
            transforms.Resize(128),
            transforms.CenterCrop(116),
            transforms.ToTensor(),            
            ]

        transforms_list = [
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
          ]
        
        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.enlarge = transforms.Compose(
                self.enlarge_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])            
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
            self.enlarge = transforms.Compose(
                self.enlarge_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        else:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])  
            self.enlarge = transforms.Compose(
                self.enlarge_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(split_map[setname], name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append( path )
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
            if self.augment:
                image_aug = self.enlarge(data)            
        else:
            image = self.transform(Image.open(data).convert('RGB'))
            if self.augment:
                image_aug = self.enlarge(Image.open(data).convert('RGB'))            
        
        if self.augment:
            return image, image_aug, label
        else:
            return image, image, label

