import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import se_resnet
import dataset
from PIL import Image
import shutil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--eval_imgpath', type=str, default='../valid-rotate/',
                    help='path to evalation dataset list file')
parser.add_argument('--eval_list', type=str, default='valid.txt', help='path to evalation dataset list file')
parser.add_argument('--net', type=str, default='seresnet50', help='net type')
parser.add_argument('--height', type=int, default=224,
                    help='input height of network')
parser.add_argument('--width', type=int, default=224,
                    help='input width of network')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_path', type=str, default='./models/seresnet50/seresnet50_final.pth', help='path to save models')
parser.add_argument('--class_num', type=int, default=4, help='the total classes')
args = parser.parse_args()

def img_label_dic(filepath):
    f1 = open(filepath,'r',encoding='utf-8')
    file_list1 = f1.readlines()
    dic_img_label = {}
    for line in file_list1:
        line_splits = line.strip('\r\n').strip().split()
        img = line_splits[0]
        label = line_splits[1]
        dic_img_label[img] = int(label)
    f1.close()
    return dic_img_label

to_bgr_transform = transforms.Lambda(lambda x: x[[2, 1, 0]])
transform = transforms.Compose([
    transforms.ToPILImage(),
    dataset.PadImage(),
    transforms.Resize([args.height, args.width],interpolation=3), transforms.ToTensor(),
    to_bgr_transform,
    transforms.Normalize((0.5,), (0.5,))
])
transform_crop = transforms.Compose([
    transforms.ToPILImage(),
    dataset.CropPadImage(),
    transforms.Resize([args.height, args.width],interpolation=3), transforms.ToTensor(),
    to_bgr_transform,
    transforms.Normalize((0.5,), (0.5,))
])
torch.backends.cudnn.benchmark = True


def run_on_opencv_block(img):
    im = transform(img).to(device)
    im = im.reshape(1, *im.size())
    net.eval()
    outputs = net(im)
    _, preds = outputs.max(1)
    label = preds[0].item()
    return label

def run_on_opencv_imgbatch(img):
    images = [transform_crop(img) for i in range(3)]
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    images = images.to(device)
    net.eval()
    outputs = net(images)
    preds = torch.softmax(outputs, 1)
    mea = torch.mean(preds, 0)
    ind = mea.argmax()
    label = ind.item()
    return label

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = se_resnet.se_resnet50(num_classes=args.class_num).to(device)
    # input = torch.randn(8, 3, 214, 214).to(device)
    # outputs = net(input)
    # print(outputs.size())
    net.load_state_dict(torch.load(args.model_path))
    dic_img = img_label_dic(args.eval_list)
    total = 0
    correct = 0
    for k,v in dic_img.items():
        print(k)
        imgpath = args.eval_imgpath + k
        if not os.path.exists(imgpath):
            print('not exist image {}'.format(k))
            continue
        pred = run_on_opencv_imgbatch(cv2.imread(imgpath))
        labels = v
        total += 1
        if pred == labels:
            correct += 1
        else:
            #name = k[:k.rfind('.')] + '_{0}.jpg'.format(pred)
            #new = '../wrong/' + name
            #shutil.copy(imgpath, new)
            print('{}->{}'.format(k, pred))
    print(correct * 1.0 / total)



