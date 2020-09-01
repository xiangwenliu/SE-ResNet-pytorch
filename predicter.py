import os
import torch
import torchvision.transforms as transforms
import se_resnet
from PIL import Image,ImageOps
import cv2
import random
from math import atan, pi, sin, cos, fabs, radians

# parser = argparse.ArgumentParser()
# parser.add_argument('--eval_imgpath', type=str, default='../valid-data/',
#                     help='path to evalation dataset list file')
# parser.add_argument('--eval_list', type=str, default='valid.txt', help='path to evalation dataset list file')
# parser.add_argument('-net', type=str, default='seresnet34', help='net type')
# parser.add_argument('--height', type=int, default=224,
#                     help='input height of network')
# parser.add_argument('--width', type=int, default=224,
#                     help='input width of network')
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--model_path', type=str, default='./models/seresnet50/seresnet50_final.pth', help='path to save models')



class PadImage(object):
    def __init__(self):
        pass

    def __call__(self, img):
        w, h = img.size
        pad = abs(w - h) // 2
        padding = (0, pad, 0, pad)
        if w < h:
            padding = (pad, 0, pad, 0)
        pad_image = ImageOps.expand(img, padding)
        return pad_image

class CropPadImage(object):
    def __init__(self):
        pass

    def __call__(self, img):
        w, h = img.size
        pad_image = img
        min_value = min(w, h) // 15
        start_height = random.randint(0, min_value)
        start_width = random.randint(0, min_value)
        end_height = int(pad_image.size[1] - random.randint(0, min_value))
        end_width = int(pad_image.size[0] - random.randint(0, min_value))
        img = pad_image.crop((start_width, start_height, end_width, end_height))
        w, h = img.size
        pad = abs(w - h) // 2
        padding = (0, pad, 0, pad)
        if w < h:
            padding = (pad, 0, pad, 0)
        img = ImageOps.expand(img, padding)
        return img

class Demo(object):
    def __init__(
            self,
            model_path='./models/seresnet50/seresnet50_final.pth',
            img_size=224,
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('devices=',self.device)
        self.net = se_resnet.se_resnet50(num_classes=4).to(self.device)
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        to_bgr_transform = transforms.Lambda(lambda x: x[[2, 1, 0]])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            PadImage(),
            transforms.Resize([self.img_size, self.img_size], interpolation=3),
            transforms.ToTensor(),
            # to_bgr_transform,
            transforms.Normalize((0.5,), (0.5,))])
        #for one image crop to a batch
        self.transform_crop = transforms.Compose([
            transforms.ToPILImage(),
            CropPadImage(),
            transforms.Resize([self.img_size, self.img_size], interpolation=3),
            transforms.ToTensor(),
            to_bgr_transform,
            transforms.Normalize((0.5,), (0.5,))])

    def run_on_opencv_image(self, img):
        im = self.transform(img).to(self.device)
        im = im.reshape(1, *im.size())
        self.net.eval()
        outputs = self.net(im)
        _, preds = outputs.max(1)
        label = preds[0].item()
        return label

    def run_on_opencv_imgbatch(self, img):
        images = [self.transform_crop(img) for i in range(3)]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        images = images.to(self.device)
        # print('----image.size()=',images.size())
        # im = self.transform_crop(img).to(self.device)
        # im = im.reshape(1, *im.size())
        self.net.eval()
        outputs = self.net(images)
        # _, preds = outputs.max(1)
        # label = preds[0].item()
        preds = torch.softmax(outputs,1)
        # mer = torch.sum(preds,1)
        mea = torch.mean(preds,0)
        ind = mea.argmax()
        label = ind.item()
        return label


# torch.backends.cudnn.benchmark = True



block_root = '../valid-data/'

if __name__ == '__main__':
    demo = Demo()
    total = 0
    correct = 0
    for k in os.listdir(block_root):
        imgpath = block_root + k
        if not os.path.exists(imgpath):
            print('not exist image {}'.format(k))
            continue
        # img = Image.open(imgpath).convert('RGB')
        img = cv2.imread(imgpath)
        #label = demo.run_on_opencv_image(img)
        label = demo.run_on_opencv_imgbatch(img)
        print('{} -> {}'.format(k,label))
        # break


        


