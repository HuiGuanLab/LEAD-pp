from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import argparse
import time
import json
import shutil
import os
import pandas as pd
import torch
import torch.nn as nn
from utils import set_seed
from dataset import MyDataSet

from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(description='linear probing')

parser.add_argument('-t', '--task', default='bird', help='Task Name: bird or car or aircraft')

parser.add_argument('--arch', default='resnet50')
parser.add_argument('--lr', '--learning-rate', default=30., type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')

parser.add_argument('--num-classes', default=200, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')

# utils
parser.add_argument('--root', default='bird/', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--resume', default='checkpoints/result_bird/model_last.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoints', default='checkpoints_linear/linear_bird/', type=str, metavar='PATH', help='path to save')

args = parser.parse_args()  # running in ipynb
print(args)

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

def train(train_loader, model, criterion, optimizer, epoch):
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

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

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

def validate(val_loader, model, criterion, args):
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

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

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
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # set tensor_board
    tbwriter = SummaryWriter('tensorboard/'+args.checkpoints)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # print(parameters)
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    # load model if resume
    epoch_start = 1
    model = model.cuda()

    # from moco trained by unsupervised
    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'): 
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)

        print('Loaded from: {}'.format(args.resume))

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    # dump args
    with open(args.checkpoints + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    best_acc1 = 0.0
    best_acc5 = 0.0
    for epoch in range(epoch_start, args.epochs + 1):

        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        tbwriter.add_scalar('Train/loss Total', train_loss, epoch)
        results['train_loss'].append(train_loss)
        acc1, acc5 = validate(test_loader, model, criterion, args)
        tbwriter.add_scalar('Test/acc1', acc1, epoch)
        results['test_acc@1'].append(acc1.cpu().numpy())
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.checkpoints + '/log.csv', index_label='epoch')
        is_best = float(acc1.cpu().numpy()) > best_acc1
        if is_best:
            best_acc1 = acc1.cpu().item()
            best_acc5 = acc5.cpu().item()
        torch.save({'best_acc1':best_acc1,'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.checkpoints + '/model_last.pth')
        if is_best :
            torch.save({'best_acc1':best_acc1,'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.checkpoints + '/model_best.pth')

    print('==> Finished training.')
    print(f'Best Test Acc@1: {best_acc1:.2f}%')
    print(f'Best Test Acc@5: {best_acc5:.2f}%')

    tbwriter.close()

if __name__ == '__main__':
    main()








