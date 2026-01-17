import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
from numpy import ndarray
from skimage import measure
import cv2
from DDT import DDT_fit
import os.path as osp
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed')
parser.add_argument('--root', type=str, default='../bird', help="the train data folder path.")
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='../resnet50-19c8e357.pth', type=str, metavar='PATH', help='path to pretrained checkpoint (default: none)')

parser.add_argument('--batch-size', default=2, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--num_workers', default=1, type=int, help='')

parser.add_argument("--save-vec", type=str, default="./DDT_result/bird/vec_res50_bird.npy", help="cuda device to run")
parser.add_argument("--save-mean-tensor", type=str, default="./DDT_result/bird/mean_tensor_bird.pth", help="cuda device to run")

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(args.seed)
    train_transform = transforms.Compose([
    transforms.Resize((512, 512), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(args.root,train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = models.resnet50().cuda()
    model.avgpool=nn.Identity()
    model.fc = nn.Identity()
    

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint
        new_state_dict={}
        for k,v in state_dict.items():
                if not k.startswith('fc'):
                    new_state_dict[k] = v
        msg = model.load_state_dict(new_state_dict,strict= False)
        print(msg.missing_keys)
        print("=> loaded pre-trained model '{}'".format(args.pretrained))

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer  
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'): 
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)

        print('Loaded from: {}'.format(args.resume))
    
    trans_vec, descriptors_mean_tensor = DDT_fit(model,train_loader,args.batch_size)
    np.save(args.save_vec,trans_vec)
    torch.save(descriptors_mean_tensor,args.save_mean_tensor)
    print("------------successful save vec and mean_tensor---------------")



if __name__=="__main__":
    main()


