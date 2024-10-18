import random
import PIL
import torch
from torchvision import transforms
import numpy as np


class AddSaltPepperNoise(object):

    def __init__(self, density=0.0, prob=0.5):
        self.density = density
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            img = np.array(img)
            height, width, channel = img.shape
            density = self.density
            s_d = 1 - density
            mask = np.random.choice((0, 1, 2), size=(height, width, 1), p=[density / 2.0, density / 2.0, s_d])
            mask = np.repeat(mask, channel, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
            img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
    ):
        self.source = source
        # transform
        self.transform_noise = transforms.Compose([
            # transforms.RandomChoice(transforms),
            # AddSaltPepperNoise(0.05, 1),
            # AddGaussianNoise(std=0.05),
            # transforms.GaussianBlur(3),
            # transforms.RandomHorizontalFlip(p=1),
            # transforms.RandomRotation(10),
            transforms.RandomAffine(10, (0.1, 0.1), (0.9, 1.1), 10)
        ])


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        item = self.source[idx]

        item["image"] = self.transform_noise(item["image"])
        return item
