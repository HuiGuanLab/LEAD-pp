from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import argparse
import time
import json
import shutil
import math
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision.datasets.folder import default_loader
from resnet import resnet50
from models import MoCo_three_gradcam,Classifier


from dataset import MyDataSet_DDT

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

#torch.cuda.set_device(1)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description='Train MoCo on CUB200')

parser.add_argument('-a', '--arch', default='resnet50')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=30., type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')

parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed')
parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--num_classes', default=200, type=int, metavar='N', help='mini-batch size')######
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')


# moco specific configs:
parser.add_argument('--moco-dim', default=512, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=65536, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.9999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.2, type=float, help='softmax temperature')


# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--test', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--crop-root', default='./DDT/crop_dataset/bird_crop', type=str, metavar='PATH',help='path to crop dataset')######
parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

args = parser.parse_args()  # running in ipynb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
# === Best Acc tracker (copy-paste) ===========================================
class BestTop1WithTop5:
    """Track the best Acc@1 and the corresponding Acc@5 (and epoch)."""
    def __init__(self):
        self.best_top1 = 0.0
        self.top5_at_best = 0.0
        self.epoch = 0

    def update(self, top1, top5, epoch: int):
        # accept float or 0-dim tensors
        t1 = float(top1) if hasattr(top1, "item") else top1
        t5 = float(top5) if hasattr(top5, "item") else top5
        if t1 > self.best_top1:
            self.best_top1 = t1
            self.top5_at_best = t5
            self.epoch = epoch

    def summary(self) -> str:
        return (f"Best Acc@1: {self.best_top1:.3f}% (epoch {self.epoch}) | "
                f"Acc@5 at best Acc@1: {self.top5_at_best:.3f}%")
# ============================================================================



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


print(args)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    # normalize])
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data prepare
crop_train = os.path.join(args.crop_root, "train") if os.path.isdir(os.path.join(args.crop_root, "train")) else args.crop_root
crop_test  = os.path.join(args.crop_root, "test")  if os.path.isdir(os.path.join(args.crop_root, "test"))  else args.crop_root

train_data = MyDataSet_DDT(crop_root=crop_train, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

test_data  = MyDataSet_DDT(crop_root=crop_test,  transform=test_transform)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


# create model
print("=> creating model '{}'".format(args.arch))
base_encoder = resnet50()
model = MoCo_three_gradcam(
        base_encoder = base_encoder,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        mlp=True
    ).cuda()

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
classifier = Classifier(2048, args.num_classes).cuda()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(train_loader, model, classifier, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    classifier.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

        # compute output
        featmap_cam = model.inference(images)
        output = classifier(featmap_cam)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    return losses.avg

def validate(val_loader, model, classifier, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # compute output
            featmap_cam = model.inference(images)
            output = classifier(featmap_cam)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return float(top1.avg), float(top5.avg)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# set random seed
set_seed(args.seed)
# set tensor_board
tbwriter = SummaryWriter('tensorboard/'+args.results_dir)
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(classifier.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
# load model if resume
epoch_start = 1
model = model.cuda()

# from moco trained by unsupervised
if args.resume != '':
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        #if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):   
        if k.startswith('module.'): 
            # remove prefix
            #state_dict[k[len("encoder_q."):]] = state_dict[k]
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]


    msg = model.load_state_dict(state_dict, strict=False)
    print(msg.missing_keys)

    print('Loaded from: {}'.format(args.resume))


# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

best_acc1 = 0.0
tracker = BestTop1WithTop5()

for epoch in range(epoch_start, args.epochs + 1):

    adjust_learning_rate(optimizer, epoch, args)
    #StepLR.step()
        # train for one epoch
    train_loss = train(train_loader, model, classifier, criterion, optimizer, epoch)
    tbwriter.add_scalar('Train/loss Total', train_loss, epoch)
    results['train_loss'].append(train_loss)
    
        # evaluate on validation set
    acc1, acc5 = validate(test_loader, model, classifier, criterion, args)
    tbwriter.add_scalar('Test/acc1', acc1, epoch)
    results['test_acc@1'].append(acc1)

    tracker.update(acc1, acc5, epoch)

     # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # remember best acc@1 and save checkpoint
    # save model
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)    
    torch.save({'best_acc1':best_acc1,'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
    if is_best :
        torch.save({'best_acc1':best_acc1,'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_best.pth')
            
tbwriter.close()

print("\n================== Linear Probing Summary ==================")
print(tracker.summary())
print("============================================================\n")







