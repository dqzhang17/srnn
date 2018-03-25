import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable as Var
from srnn_datasets import SrnnDataset
from srnn import SRnn

import sys
sys.path.append('.')
import pretrainedmodels

use_gpu = True

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training for SRnn')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--basenet', '-a', metavar='BASENET', default='resnet18',
                    choices=model_names,
                    help='basenet: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--basenet_pretrained', '-r', dest='basenet_pretrained', action='store_true',
                    help='use pretrained basenet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained srnn model with base CNN (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

parser.add_argument('--train_cnn', '-c', dest='train_cnn', action='store_true',
                    help='train the base cnn with srnn')
parser.add_argument('--volatile_basenet', '-v', dest='volatile_basenet', action='store_true',
                    help='set basenet to volatile (for GPU memory saving)')
parser.add_argument('--mode', default=0, type=int,
                    help='algorithm mode : 0 - single scale inference, 1 - post-softmax scale ensemble, \
                    2 - pre-softmax scale ensemble (srnn with identity state transition), \
                    3 - vanilla srnn, 4 - half-GRU srnn, 5 - GRU srnn ')
parser.add_argument('--single_scale', default=0, type=int,
                    help='scale index for the single scale mode - {0, 1, 2, ...}')
parser.add_argument('--lr_drop', default=30, type=int,
                    help='lr drop interval (epochs)')
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.train_cnn or args.evaluate:
        args.volatile_basenet = False

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model = SRnn(args.basenet, pretrained_base=args.basenet_pretrained, train_cnn=args.train_cnn, mode=args.mode, single_scale=args.single_scale)
    model_config = ModelConfig(model.get_basenet())

    # loading pretrained model if assigned
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model '{}'".format(args.pretrained))
            pretrained = torch.load(args.pretrained)
            model.load_state_dict(pretrained['state_dict'])
            print("=> loaded pretrained '{}'".format(args.pretrained))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

# basenet
    basenet = None
    if args.volatile_basenet:
        basenet = model.get_and_remove_basenet() # take out basenet

# gpu or distributed processing
    if not args.distributed:
        if args.basenet.startswith('alexnet') or args.basenet.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if use_gpu: model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda() if use_gpu else model
            if basenet:
                basenet = torch.nn.DataParallel(basenet).cuda() if use_gpu else basenet
    else:
        if use_gpu:
            model.cuda()
            if basenet: basenet.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        if basenet: torch.nn.parallel.DistributedDataParallel(basenet)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()

# selectively pick parameters
    net_parameters = []
    names = []
    for name,para in model.named_parameters():
        if not args.train_cnn:   # fix base cnn
            if not name.startswith('base_net'):
                net_parameters.append(para)
                names.append(name)
            else: para.require_grad = False
        else:
            net_parameters.append(para)
            names.append(name)

    optimizer = torch.optim.SGD(net_parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = SrnnDataset(traindir,'train',model_config)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=use_gpu, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        SrnnDataset(valdir, 'val', model_config),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_gpu)

    if args.evaluate:
        validate(val_loader, model, basenet, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, basenet, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, basenet, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'basenet': args.basenet,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, basenet, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    if basenet:
        basenet.eval()   # always set to evaluation if basenet is volatile

    end = time.time()
    for i, sample in enumerate(train_loader):
        input = sample['image'];
        target = sample['target']
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True) if use_gpu else target
        input_var = [Var(inp) for inp in input]
        target_var = Var(target)

        # compute base output
        if basenet:
            for idx in range(len(input_var)):
                input_var[idx].volatile = True
                tmp = basenet(input_var[idx])
                input_var[idx] = Var(tmp.data)  # re-pack

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input[0].size(0))
        top1.update(prec1[0], input[0].size(0))
        top5.update(prec5[0], input[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)

def validate(val_loader, model, basenet, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if basenet:
        basenet.eval()   # always set to evaluation if basenet is volatile

    end = time.time()
    for i, sample in enumerate(val_loader):
        input = sample['image'];
        target = sample['target']
        target = target.cuda(async=True) if use_gpu else target
        input_var = [Var(inp, volatile=True) for inp in input]
        target_var = Var(target, volatile=True)

        # compute base output
        if basenet:
            for idx in range(len(input_var)):
                input_var[idx].volatile = True
                tmp = basenet(input_var[idx])
                input_var[idx] = Var(tmp.data)  # re-pack

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input[0].size(0))
        top1.update(prec1[0], input[0].size(0))
        top5.update(prec5[0], input[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5), flush=True)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5), flush=True)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class ModelConfig(object):
    def __init__(self, model):
        self.mean = model.mean
        self.std = model.std
        self.input_size = model.input_size # nn input size

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every args.lr_drop epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_drop))
    print('lr= '+str(lr), flush=True)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
