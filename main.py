from datetime import datetime
from functools import partial
from sched import scheduler
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import argparse
import json
import math
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.multiprocessing as mp
from resnet import resnet50
import clip

from utils import set_seed,get_lr,Similarity,GaussianBlur
from dataset import MyPair_entropy_DDT_four
# from ViT import vit_large_patch16_224 as ViT
# from ViT import vit_base_patch16_224 as ViT
from model_new import MoCo_gradcam_KL

# from tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Train MoCo on CUB200')

parser.add_argument('-a', '--arch', default='clip')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', default=True, help='use cosine lr schedule')

parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=512, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=65536, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.9999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.2, type=float, help='softmax temperature')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='./resnet50-19c8e357.pth', type=str, metavar='PATH', help='path to pretrained checkpoint (default: none)')

parser.add_argument('--root', default='bird/', type=str, metavar='PATH', help='path to latest checkpoint (default: /data/datasets/CUB_200/CUB_200_2011)')
parser.add_argument('--crop_root', default='./DDT/crop_dataset/bird_crop', type=str, metavar='PATH', help='path to crop dataset')
parser.add_argument('--results-dir', default='checkpoints/LEAD++_bird/', type=str, metavar='PATH', help='path to cache (default: none)')

parser.add_argument('--port', default=10002, type=int, help='')
parser.add_argument('--world-size', default=2, type=int, help='')
parser.add_argument('--num_workers', default=16, type=int, help='')
parser.add_argument('--num-classes', default=200, type=int, metavar='N', help='Total number of categories')

parser.add_argument('--alpha', default=0.5, type=float, help='contrastive loss weight')
parser.add_argument('--beta', default=10, type=float, help='distill loss weight')
parser.add_argument('--text-load', default='text_description_tensor/bird_text_tensor.pt', type=str, help='text of clip feature')
parser.add_argument('--temperature', default=0.02, type=float, help='softmax temperature')

args = parser.parse_args()

if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

def main():
    # args & cfg
    set_seed(args.seed)

    # world_size = torch.cuda.device_count()
    world_size = args.world_size
    print('GPUs on this node:', world_size)

    # spawn
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

def main_worker(rank, world_size, args):
    print('==> Start rank:', rank)

    local_rank = rank % 8
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{args.port}',
        world_size=world_size,
        rank=rank
    )

    # build data loader
    bsz_gpu = int(args.batch_size / args.world_size)
    print('batch_size per gpu:', bsz_gpu)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        # transforms.RandomResizedCrop((448,448), scale=(0.2, 1.0)),    
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    traindir = os.path.join(args.root, "train")
    crop_train = os.path.join(args.crop_root, "train") if os.path.isdir(os.path.join(args.crop_root, "train")) else args.crop_root
    train_data = MyPair_entropy_DDT_four(img_root=traindir, transform=train_transform, crop_root=crop_train,strict_crop=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=bsz_gpu, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    device = torch.device(f'cuda:{rank}')
    model_t, _ = clip.load("ViT-B/16", device=device)
    model_t.float()
    model_t_dim = model_t.ln_final.weight.shape[0]
    model_t = torch.nn.parallel.DistributedDataParallel(model_t, device_ids=[local_rank], broadcast_buffers=False)
    model_t.eval()

    base_encoder = resnet50()
    model = MoCo_gradcam_KL(
        base_encoder=base_encoder,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        teacher_dim=model_t_dim,
        mlp=True
    ).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    criterion1 = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion2 = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion3 = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_distill = torch.nn.KLDivLoss(reduction='none').cuda()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('fc'):
                    new_state_dict["module.encoder_q." + k] = v
                    new_state_dict["module.encoder_k." + k] = v
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(msg.missing_keys)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


    test_tensor = torch.load(args.text_load, map_location='cpu').cuda()
    epoch_start = 1
    # Start training
    print("==> Start training...")
    for epoch in range(epoch_start, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(model, model_t, train_loader, criterion1, criterion2, criterion3, criterion_distill, optimizer, epoch, test_tensor, args)
        # tbwriter.add_scalar('Train/loss Total', train_loss, epoch)

    # save the last model; master process
    if not os.path.exists(args.results_dir) and rank == 0:
        os.mkdir(args.results_dir)
    if rank == 0:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}, args.results_dir + '/model_last.pth')

# train for one epoch
def train(net, model_t, data_loader, criterion1, criterion2, criterion3, criterion_distill, train_optimizer, epoch, test_tensor, args):
    net.train()
    model_t.eval()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for im_1, im_2, im_3, im_4 in train_bar:
        im_1, im_2, im_3, im_4 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True), im_3.cuda(non_blocking=True), im_4.cuda(non_blocking=True)

        output1, target1, _, _ = net(im_1, im_2, is_first=True, is_second=False)
        output2, target2, _, _ = net(im_1, im_4, is_first=False, is_second=True)
        output3, target3, f_s, featmap = net(im_3, im_4, is_first=False, is_second=False)

        f_t = model_t.module.encode_image(im_3)
        f_t = nn.functional.normalize(f_t, dim=1, p=2.0)

        test_tensor = test_tensor.to(torch.float32)
        similarity_t = f_t @ test_tensor.T
        similarity_t /= args.temperature
        similarity_t_softmax = F.softmax(similarity_t, dim=1)
        similarity_s = f_s @ test_tensor.T
        similarity_s /= args.temperature
        similarity_s_softmax = F.softmax(similarity_s, dim=1)
        loss_distill = criterion_distill(similarity_s_softmax.log(), similarity_t_softmax)

        loss1 = criterion1(output1, target1)
        loss2 = criterion2(output2, target2)
        loss3 = criterion3(output3, target3)

        entropy = -torch.sum(similarity_t_softmax * torch.log(similarity_t_softmax + 1e-10), dim=1)
        max_entropy = torch.log(torch.tensor(args.num_classes, dtype=torch.float32, device=entropy.device))
        entropy = entropy / max_entropy

        weights_contrastive = torch.tensor([p * p for p in entropy], device=entropy.device)
        weights_distill = torch.tensor([1 - q for q in weights_contrastive], device=entropy.device)

        loss_distill = loss_distill.mean(dim=1) * weights_distill * args.beta
        loss_distill = loss_distill.sum()
        loss1 = args.alpha * loss1 * weights_contrastive
        loss1 = torch.mean(loss1)
        loss2 = args.alpha * loss2 * weights_contrastive
        loss2 = torch.mean(loss2)
        loss3 = args.alpha * loss3 * weights_contrastive
        loss3 = torch.mean(loss3)

        grad_wrt_act1 = torch.autograd.grad(outputs=loss3, inputs=featmap,
                                            grad_outputs=torch.ones_like(loss3), retain_graph=True,
                                            allow_unused=True)[0]

        gradcam = torch.relu((featmap * grad_wrt_act1).sum(dim=1))
        att_max = net.module.max_mask(featmap)
        loss_cam = F.kl_div(att_max.softmax(dim=-1).log(), gradcam.softmax(dim=-1),
                            reduction='sum')
        loss_cl = loss1 + 0.1 * loss2 + loss3 + loss_cam * 0.001 * args.alpha
        loss = loss_cl + loss_distill

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.10f}, Loss: {:.4f}'.format(epoch, args.epochs, get_lr(train_optimizer), total_loss / total_num))

    return total_loss / total_num

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()