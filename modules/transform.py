import torchvision
import cv2
import torch
import random
import numpy as np


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class Tran_Replace:
    def __init__(self, query, replace_prob=0.5, s=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], blur=False):
        self.replace_prob=replace_prob
        self.query=query
        self.transform = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.transform.append(GaussianBlur(kernel_size=23))
        # self.transform.append(torchvision.transforms.ToTensor())
        if mean and std:
            self.transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.transform = torchvision.transforms.Compose(self.transform)

    def test_replace(self,x):
        return x, torch.ones(x.size(0))

    def replace(self,x):
        #判定是否替换
        x0=x
        num_sample=len(self.query)
        assert(num_sample==len(self.query[0]))
        flag=torch.ones(x0.size(0))
        for index in range(x0.size(0)):
            if torch.rand(1)<self.replace_prob:
                related_positive_indices = [i for i in range(num_sample) if self.query[index][i]==1]
                related_negative_indices = [i for i in range(num_sample) if self.query[index][i]==-1]
                # related_indices = related_positive_indices
                related_indices = related_positive_indices
                # related_indices=torch.nonzero(self.query[index]).squeeze()
                # print(related_indices.size(0),related_indices)
                if(len(related_indices)==0): 
                    continue
                selected_index=random.choice(related_indices)
                x[index]=x0[selected_index]
                flag[index]=self.query[index][selected_index]
        #均等概率替换（也可must和canot分概率）
        
        #验证多维度tensor进行变换！！！！！

        return x, flag

    def __call__(self,x):
        x,flag=self.replace(x) 
        return self.transform(x),flag

class Transforms_test:
    def __init__(self, size, s=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], blur=False):
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor()
        ]
        if mean and std:
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

        
    

    def __call__(self, x):
        # return self.train_transform(x), self.train_transform(x)
        return self.test_transform(x)

class Transforms_train:
    def __init__(self, size, s=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], blur=False):
        self.train_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        self.train_transform.append(torchvision.transforms.ToTensor())
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))

        
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
    

    def __call__(self, x):
        # return self.train_transform(x), self.train_transform(x)
        return self.train_transform(x), self.train_transform(x)