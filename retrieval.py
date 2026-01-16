from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from resnet_eval import resnet50
from utils import set_seed
from dataset import MyDataSet

# from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description='retrieval')

parser.add_argument('-t', '--task', default='bird', help='Task Name: bird or car or aircraft')

parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')

# utils
parser.add_argument('--root', default='/media/data1/zzh/EAD-FFAB/EAD/dog-120', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--resume', default='/media/data1/zzh/LEAD/checkpoints/result_dog/model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


args = parser.parse_args() 
print(args)



def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        data = []
        label = []
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True) # [batch_size, 3, 224, 224]
            target = target.cuda(non_blocking=True) # [batch_size]

            # compute output
            output = model(images) # [batch_size, hidden_dim]
            output = nn.functional.normalize(output, dim=1)
            data.append(output)
            label.append(target)
        
        data = torch.cat(data, dim=0)
        label = torch.cat(label, dim=0)

        topN1 = []
        topN5 = []
        MAP = []
        for j in range(data.size(0)):
            query_feat = data[j, :]
            query_label = label[j].item()

            dict = data[torch.arange(data.size(0)) != j]
            sim_label = label[torch.arange(label.size(0)) != j]

            similarity = torch.mv(dict, query_feat)

            table = torch.zeros(similarity.size(0), 2)
            table[:, 0] = similarity
            table[:, 1] = sim_label
            table = table.cpu().detach().numpy()

            index = np.argsort(table[:, 0])[::-1]

            T = table[index]
            #top-1
            if T[0,1] == query_label:
                topN1.append(1)
            else:
                topN1.append(0)
            #top-5
            if np.sum(T[:5, -1] == query_label) > 0:
                topN5.append(1)
            else:
                topN5.append(0)

            #mAP
            check = np.where(T[:, 1] == query_label)
            check = check[0]
            AP = 0
            for k in range(len(check)):
                temp = (k+1)/(check[k]+1)
                AP = AP + temp
            AP = AP/(len(check))
            MAP.append(AP)

        top1 = np.mean(topN1)
        top5 = np.mean(topN5)
        mAP = np.mean(MAP)

        # TODO: this should also be done with the ProgressMeter
        print(' * Top@1 {top1:.4f} Top@5 {top5:.4f} mAP {mAP:.4f}'
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1, top5, mAP


def main():
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # data prepare
    traindir = os.path.join(args.root, "train")
    train_data = MyDataSet(img_root = traindir,transform = train_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    testdir = os.path.join(args.root, "test")
    test_data = MyDataSet(img_root = testdir,transform = test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


    # create model
    model = resnet50(num_classes = 512)
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc
    )
    model = model.cuda()

    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            #if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):   
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'): 
                # remove prefix
                #state_dict[k[len("encoder_q."):]] = state_dict[k]
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]


        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)

        print('Loaded from: {}'.format(args.resume))

    top1, top5, mAP = validate(test_loader,model)

if __name__ == '__main__':
    main()








