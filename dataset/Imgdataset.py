
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
img_path="dataset/repre_vae.npz"
class ImgDataset(Dataset):
    def __init__(self,path,return_factor=False,return_index=False):
        self.return_factor=return_factor
        self.return_index=return_index
        self.data_tensor = np.load(img_path)["repre"]
        self.indices=np.load(path)["indices"].astype(np.int32)
        self.labels=np.load(path)["labels"].astype(np.int32)
        if self.return_factor:
            self.factors=np.load(path)["factors"].astype(np.int32)
    def __getitem__(self, index):
        imgs=self.data_tensor[self.indices[index]]
        labels=np.eye(imgs.shape[1])[self.labels[index]]
        if self.return_factor:
            return torch.from_numpy(imgs).float(),torch.from_numpy(labels).float(),self.factors[index]
        if self.return_index:
            return torch.from_numpy(imgs).float(),torch.from_numpy(labels).float(),self.indices[index]
        return torch.from_numpy(imgs).float(),torch.from_numpy(labels).float()
        
    def __len__(self):
        return self.labels.shape[0]

if __name__=="__main__":
    dataset=ImgDataset()
    img,lb=dataset[1]
    print(img)
    print(lb)