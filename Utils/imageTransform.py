from torchvision import transforms as T
import torch
import random
from torchvision.transforms import functional as TF

"""
# 训练时图像随机裁剪的做法:
    #1: 先进行保持图片横纵比不变的压缩后再进行随机裁剪:Resize_and_RandomCrop.
    #2: 直接在原图上做随机裁剪
"""


def pad_if_smaller(img, size, fill=0):
    pass


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob:float):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, flip_prob:float):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
        return image


class Resize(object):
    """
    # 如果 imgSize 是 int,则图像的长、宽中较小的那一边将与此数字匹配，同时保持图像的纵横比
    # 假设原始输入图像大小为(3, 500, 332), imgSize=224, 则最好输出的图片大小为(3, 337, 224)
    # 如果在实际应用中需要保持横纵比的话，需要结合crop去做图像裁剪比较好
    """
    def __init__(self, imgSize):
        self.imgSize = imgSize

    def __call__(self, image):
        image = TF.resize(image, self.imgSize)
        return image


class Resize_and_RandomCrop(object):
    """
    # 先对图像进行保持横纵比的压缩，在进行随机裁剪到指定大小
    """
    def __init__(self, size:int):
        self.size = size

    def __call__(self, image):
        image = Resize(self.size)(image)
        # 获取随机裁剪的参数，这操作在分割任务中对image,label进行同步裁剪操作是一个不错的技巧
        # Pytorch官方说明:https://pytorch.org/vision/0.12/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        return image


class RandomCrop(object):
    """
    # 直接在原图上做随机裁剪
    """
    def __init__(self, size:int):
        self.size = size

    def __call__(self, image):
        image = Resize(self.size)(image)
        # 获取随机裁剪的参数，这操作在分割任务中对image,label进行同步裁剪操作是一个不错的技巧
        # Pytorch官方说明:https://pytorch.org/vision/0.12/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        return image

class ToTensor(object):
    def __call__(self, image):
        image = TF.to_tensor(image)
        return image
