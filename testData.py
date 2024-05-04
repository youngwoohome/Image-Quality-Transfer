import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate


def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))

    if batch == []:
        return None

    return default_collate(batch)


class supervisedIQT_INF(Dataset):
    def __init__(self, lr_file):
        self.lr_file = lr_file

        self.mean_lr = 271.64814106698583
        self.std_lr = 377.117173547721

        self.patch_size = 32
        self.overlap = 16
        self.ratio = 0.1
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size

        self.lr_idx = []
        low,high = 8,248
        self.lr_data = nib.load(self.lr_file).get_fdata()[low:high, low:high, low:high]
        for i in range(0,self.lr_data.shape[0]-self.patch_size+1,self.overlap):
            for j in range(0,self.lr_data.shape[1]-self.patch_size+1,self.overlap):
                for k in range(0,self.lr_data.shape[2]-self.patch_size+1,self.overlap):
                    self.lr_idx.append([i,j,k])

    def __len__(self):
        return len(self.lr_idx)

    def normalize(self, img): # transform 3D array to tensor

        image_torch = torch.FloatTensor(img)
        image_torch = (image_torch - self.mean_lr)/self.std_lr
        return image_torch
    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283]

        return hyp_norm

    def __getitem__(self, idx):

        self.lr = self.lr_idx[idx]

        self.lr = self.lr_data[self.lr[0]:self.lr[0]+self.patch_size, self.lr[1]:self.lr[1]+self.patch_size, self.lr[2]:self.lr[2]+self.patch_size]
        self.lr = torch.tensor(self.lr.astype(np.float32))
        self.img_shape = self.lr.shape

        non_zero = np.count_nonzero(self.lr)
        non_zero_proportion = (non_zero/self.total_voxel)
        #print(non_zero_proportion)
        if (non_zero_proportion < self.ratio):
            return None  #self.__getitem__(idx+1)

        self.lr = self.normalize(self.lr)

        sample_lr = torch.unsqueeze(self.lr, 0)

        return [sample_lr, torch.tensor(self.lr_idx[idx])]

