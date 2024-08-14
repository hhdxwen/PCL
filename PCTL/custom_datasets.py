import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset

class ImageNetData():
    def __init__(self, path, seed, split_ratio=0.2, transform=None):
        np.random.seed(seed)
        self.imagenet = datasets.ImageFolder(root=path,transform=transform)
        self.random_index = np.random.permutation(len(self.imagenet))
        split_index = int(split_ratio * len(self.imagenet))
        self.train_indices = self.random_index[split_index:]
        self.test_indices = self.random_index[:split_index]
        self.data_train=[]
        self.data_test=[]
        for i in self.train_indices:
            self.data_train.append(self.imagenet[i])
        for i in self.test_indices:
            self.data_test.append(self.imagenet[i])

    def get_data(self):
        return self.data_train,self.data_test

class ImageNet(Dataset):
    def __init__(self, data, transform=None):
        self.data=data
        self.transform=transform

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        image, label = self.data[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.data)
# from torch.utils.data import Dataset, DataLoader
# from modules import transform
# import torchvision
# import numpy
# import numpy as np

# class ImageNet(Dataset):
#     def __init__(self, path, image_size, seed):
#         np.random.seed(seed)
#         self.imagenet = torchvision.datasets.ImageFolder(
#                                 root=path,
#                                 transform=transform.Transforms(size=image_size, blur=True),
#                             )
#         self.random_index = np.random.permutation(len(self.imagenet))

#     def __getitem__(self, index):
#         if isinstance(index, numpy.float64):
#             index = index.astype(numpy.int64)

#         return  self.imagenet[self.random_index[index]]

#     def __len__(self):
#         return len(self.imagenet)
