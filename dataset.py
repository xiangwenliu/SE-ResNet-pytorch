import torch
import torchvision
import random
from PIL import Image
import cv2
import numpy as np
from skimage.transform import warp, AffineTransform
from scipy.ndimage import rotate
import skimage
from scipy.ndimage import rotate
from math import atan, pi, sin, cos, fabs, radians
from PIL import ImageOps

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

class ImageAug(object):
    def __init__(self):
        pass

    def gaussian_blur(self, img):
        k_size = 2 * random.randint(0, 4) + 1
        kernel_size = (k_size, k_size)
        sigma = random.random() * 3
        img = cv2.GaussianBlur(img, kernel_size, sigma)
        return img

    def Noise(self, img):
        mode = ['gaussian', 'localvar', 'poisson',
                'salt', 'pepper', 's&p', 'speckle']
        current_mode = 'salt'
        img = img.astype('float32') - img.min()
        img = img / img.max()
        img = skimage.util.random_noise(img, mode=current_mode)
        img = np.array(255 * img, dtype='uint8')
        return img

    def pad_random_crop_img(self, img):
        w, h = img.size
        # pad = abs(w - h) // 2
        # padding = (0, pad, 0, pad)
        # if w < h:
        #     padding = (pad, 0, pad, 0)
        # pad_image = ImageOps.expand(img, padding)
        pad_image = img
        min_value = min(w, h) // 15
        start_height = random.randint(0, min_value)
        start_width = random.randint(0, min_value)
        end_height = int(pad_image.size[1] - random.randint(0, min_value))
        end_width = int(pad_image.size[0] - random.randint(0, min_value))
        img = pad_image.crop((start_width, start_height, end_width, end_height))
        return img

    def rotate_image(self, img, angle):
        h, w = img.shape[:2]
        rad = radians(angle)
        New_height = int(w * fabs(sin(rad)) + h * fabs(cos(rad)))
        New_width = int(h * fabs(sin(rad)) + w * fabs(cos(rad)))
        M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        M_rotate[0, 2] += (New_width - w) / 2
        M_rotate[1, 2] += (New_height - h) / 2
        img_rotated = cv2.warpAffine(img, M_rotate, (New_width, New_height), flags=cv2.INTER_CUBIC,
                                     borderValue=[0, 0, 0])
        return img_rotated

    def perspective_transform(self, img):
        w, h = img.shape[1], img.shape[0]
        pts1 = np.float32([[0, 0], [0, w], [h, w], [h, 0]])
        pts2 = np.float32([[random.randint(0, 3), random.randint(0, 3)],
                           [random.randint(0, 3), w - random.randint(0, 3)],
                           [h - random.randint(0, 3), w - random.randint(0, 3)],
                           [h - random.randint(0, 3), random.randint(0, 3)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h))
        return img

    def augmentation_img(self, img):
        r_angle = 0  # random.randint(0, 2)
        x_scale, y_scale = random.uniform(1. / 1.03, 1.03), random.uniform(1. / 1.03, 1.03)
        shear = random.uniform(-5 * np.pi / 210, 5 * np.pi / 210)
        x_translation, y_translation = random.randint(-2, 2), random.randint(-2, 2)
        tform = AffineTransform(scale=(x_scale, y_scale), shear=shear,
                                translation=(x_translation, y_translation))
        img_warped = warp(img, tform.inverse, output_shape=(img.shape[0], img.shape[1]))
        img = rotate(img_warped, r_angle, axes=(0, 1), order=1, reshape=False)
        img = img * (1 + random.random() * 0.1) if random.getrandbits(1) else img * (1 - random.random() * 0.1)
        img = np.clip(img*255, 0, 255).astype(np.uint8)
        return img

    def __call__(self, image, label):
        # print('----into ImageAug')
        ran = random.randint(0, 2)
        if not ran:
            # enh_bri = ImageEnhance.Brightness(image)
            # brightness = round(random.uniform(0.8, 1.2), 2)
            # image = enh_bri.enhance(brightness)
            # # plt.show()

            # # color
            # enh_col = ImageEnhance.Color(image)
            # color = round(random.uniform(0.8, 1.2), 2)
            # image = enh_col.enhance(color)

            # # contrast
            # enh_con = ImageEnhance.Contrast(image)
            # contrast = round(random.uniform(0.8, 1.2), 2)
            # image = enh_con.enhance(contrast)
            # Gaussian blur
            image = self.pad_random_crop_img(image)
            image = self.gaussian_blur(np.array(image))
            image = self.Noise(image)
            image = self.augmentation_img(image)
            # image = self.augmentation_img(np.array(image,dtype=np.int8))
            # image = self.gaussian_blur(np.array(image))

            return Image.fromarray(image), label
        else:
            return image, label


class TextLineDataset(torch.utils.data.Dataset):

    def __init__(self, rootpath=None, text_line_file=None, transform=None, target_transform=None, use_rgb=True,is_train=True):
        self.text_line_file = text_line_file
        with open(text_line_file) as fp:
            self.lines = fp.readlines()
            self.nSamples = len(self.lines)
        self.my_transform = ImageAug()
        self.transform = transform
        self.target_transform = target_transform
        self.root_path = rootpath
        self.use_rgb = use_rgb
        self.is_train = is_train

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        line_splits = self.lines[index].strip('\r\n').strip().split()
        img_path = self.root_path + line_splits[0]
        try:
            # if 'train' in self.text_line_file:
            if self.use_rgb is True:
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            print(img_path)
            return self[index + 1]
        label = line_splits[1]
        # print('---label=',label)
        label = int(label)
        if self.transform is not None:
            if self.is_train is True:
                img,label = self.my_transform(img, label)
            img = self.transform(img)



        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class ResizeNormalize(object):

    def __init__(self, img_width, img_height, img_channel):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img):
        img = np.array(img)
        if self.img_channel == 1:
            h, w = img.shape
            height = self.img_height
            width = int(w * height / h)
            if width >= self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))
            else:
                img = cv2.resize(img, (width, height))
                img_pad = np.zeros((self.img_height, self.img_width), dtype=img.dtype)
                img_pad[:height, :width] = img
                img = img_pad
        else:
            h, w, c = img.shape
            height = self.img_height
            width = int(w * height / h)
            if width >= self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))
            else:
                img = cv2.resize(img, (width, height))
                img_pad = np.zeros((self.img_height, self.img_width, c), dtype=img.dtype)
                img_pad[:height, :width, :] = img
                img = img_pad
        img = Image.fromarray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batches = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batches):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, img_height=224, img_width=224, img_channel=3):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.transform = ResizeNormalize(img_width=self.img_width, img_height=self.img_height,
                                         img_channel=self.img_channel)

    def __call__(self, batch):
        images, labels = zip(*batch)

        images = [self.transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
