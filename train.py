import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils import WarmUpLR
import se_resnet
import dataset

model_reg = {
    'seresnet18': se_resnet.se_resnet18,
    'seresnet34': se_resnet.se_resnet34,
    'seresnet50': se_resnet.se_resnet50,
    'seresnet101': se_resnet.se_resnet101,
    'seresnet152': se_resnet.se_resnet152,
}

MILESTONES = [50, 70]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataloader():
    transform = transforms.Compose([
        dataset.PadImage(),
        transforms.Resize([args.height, args.width],interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transformv = transforms.Compose([
        dataset.PadImage(),
        transforms.Resize([args.height, args.width],interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    torch.backends.cudnn.benchmark = True

    train_set = dataset.TextLineDataset(rootpath=args.train_imgpath, text_line_file=args.train_list,
                                            transform=transform, use_rgb=True,is_train=True)
    trainloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    valid_set = dataset.TextLineDataset(rootpath=args.eval_imgpath,text_line_file=args.eval_list, transform=transformv, use_rgb=True,is_train=False)
    validloader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    print('----the train length is:',len(trainloader.dataset))
    print('----the valid length is:',len(validloader.dataset))
    return trainloader, validloader


def train(epoch, trainloader):
    net.train()
    for batch_index, (images, labels) in enumerate(trainloader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        # print('----images.shape=',images.shape)
        # print('----label.shape=',labels.shape)
        images = Variable(images)
        labels = Variable(labels)
        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch_size + len(images),
            total_samples=len(trainloader.dataset)
        ))


def eval_training(validloader):
    net.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    for (images, labels) in validloader:
        images = Variable(images)
        labels = Variable(labels)
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(validloader.dataset),
        correct.float() / len(validloader.dataset)
    ))
    return correct.float() / len(validloader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_imgpath', type=str, default='../train-rotate/',
                        help='path to train dataset list file')
    parser.add_argument('--eval_imgpath', type=str, default='../valid-rotate/',
                        help='path to evalation dataset list file')
    parser.add_argument('--train_list', type=str, default='train.txt', help='path to train dataset list file')
    parser.add_argument('--eval_list', type=str, default='valid.txt', help='path to evalation dataset list file')
    parser.add_argument('--net', type=str, default='seresnet50', help='net type')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--height', type=int, default=224,
                        help='input height of network')
    parser.add_argument('--width', type=int, default=224,
                        help='input width of network')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of data loading workers')
    parser.add_argument('--model_path', type=str, default='models', help='path to save models')
    parser.add_argument('--logs_path', type=str, default='logs', help='path to save logs')
    parser.add_argument('--epochs', type=int, default=100, help='the train total epoch')
    parser.add_argument('--save_epoch', type=int, default=5, help='the step epoch save checkpoint')
    parser.add_argument('--eval_epoch', type=int, default=1, help='the step epoch for eval model')
    parser.add_argument('--class_num', type=int, default=4, help='the total classes')

    args = parser.parse_args()
    if model_reg.get(args.net) is None:
        print('Not register the request model, only support seresnet18,seresnet34,seresnet50,seresnet101,seresnet152')
        exit(0)
    net = model_reg[args.net](num_classes=args.class_num).to(device)
    # net = se_resnet.se_resnet50(num_classes=4).to(device)
    # input = torch.randn(8, 3, 214, 214).to(device)
    # outputs = net(input)
    # print(outputs.size())
    trainloader, validloader = prepare_dataloader()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    # time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    postfix = time.strftime("_%m_%d", time.localtime())
    checkpoint_path = os.path.join(args.model_path, args.net+postfix)
    logdir = checkpoint_path
    # # use tensorboard
    # logdir = os.path.join(args.model_path, args.net)
    # if not os.path.exists(args.logs_path):
    #     os.mkdir(args.logs_path)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}_{epoch}.pth')

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch, trainloader)
        acc = eval_training(validloader)
        with open('{}/train.log'.format(logdir), 'a') as f:
            f.write('[{}] Accuracy of the network on the {} validation images: {:.2%}\n'.format(
            epoch, len(validloader.dataset), acc))
        # start to save best performance model after learning rate decay to args.lr
        if epoch > MILESTONES[0] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch))
            best_acc = acc
            continue

        if not epoch % args.save_epoch:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch))
    torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch='final'))
