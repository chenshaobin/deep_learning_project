from torchvision import transforms as T
import numpy as np
import random
import torch
from torchvision.transforms import functional as TF
import PIL

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob:float):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob:float):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            target = TF.vflip(target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.float32)
        # 如果需要进行图片显示测试时，dtype=torch.float32，在训练时使用dtype=torch.int64
        return image, target


class Resize(object):
    """
    # 如果 imgSize 是 int,则图像的长、宽中较小的那一边将与此数字匹配，同时保持图像的纵横比
    # 假设原始输入图像大小为(3, 500, 332), imgSize=224, 则最好输出的图片大小为(3, 337, 224)
    # 如果在实际应用中需要保持横纵比的话，需要结合crop去做图像裁剪比较好
    """
    def __init__(self, imgSize):
        self.imgSize = imgSize

    def __call__(self, image, target):
        image = TF.resize(image, self.imgSize)
        # 在torchvision(0.9.0)之后interpolation可以使用InterpolationMode.NEAREST
        # label数据进行邻近插值
        target = TF.resize(target, self.imgSize, interpolation=PIL.Image.NEAREST)
        return image, target


class RandomCrop(object):
    """
    # 先对图像进行保持横纵比的压缩，在进行随机裁剪到指定大小
    """
    def __init__(self, size:int):
        self.size = size

    def __call__(self, image, target):
        image, target = Resize(self.size)(image, target)
        # 获取随机裁剪的参数，这操作在分割任务中对image,label进行同步裁剪操作是一个不错的技巧
        # Pytorch官方说明:https://pytorch.org/vision/0.12/generated/torchvision.transforms.RandomCrop.html#torchvision.transforms.RandomCrop
        # https://pytorch.org/vision/0.12/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        target = TF.crop(target, *crop_params)

        return image, target
