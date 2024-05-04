import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import nibabel as nib

class CustomDataset(Dataset):
    def __init__(self, configs, lr_files, hr_files, train=True):

        self.configs = configs
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.patch_size = self.configs['patch_size']
        self.ratio = self.configs['ratio']
        self.ratio_eval = self.configs['ratio_test']
        self.total_voxel = self.patch_size * self.patch_size * self.patch_size
        self.train = train
        self.is_transform = True
        self.mean_lr = configs['mean']
        self.std_lr = configs['std']

        self.files_lr = []
        self.files_hr = []

        for i in range(len(self.lr_files)):
            self.files_lr.append(self.lr_files[i])
            self.files_hr.append(self.hr_files[i])

    def __len__(self):
        return len(self.files_lr)

    # def normalize(self, img):
    #     img = 2*(img-img.min())/(img.max()-img.min())-1
    #     return img

    #z-score normalization
    def normalize(self, img):
        return (img - self.mean_lr) / self.std_lr


    def cube(self,data):

        hyp_norm = data

        if len(hyp_norm.shape)>3:
            hyp_norm = hyp_norm[:,:, 2:258, 27:283]
        else:
            hyp_norm = hyp_norm[2:258, 27:283]

        return hyp_norm

    def __getitem__(self, idx):

        self.lr = self.files_lr[idx]
        self.hr = self.lr.replace(self.configs['lr_name'],self.configs['gt_name'])

        self.lr = nib.load(self.lr)
        self.lr_affine = self.lr.affine
        self.lr = torch.tensor(self.lr.get_fdata().astype(np.float32))
        self.img_shape = self.lr.shape
        self.hr = nib.load(self.hr)
        self.hr_affine = self.hr.affine
        self.hr = torch.tensor(self.hr.get_fdata().astype(np.float32))

        #Cube
        self.lr = self.lr[14:238,14:238,14:238]
        self.hr = self.hr[14:238,14:238,14:238]

        random_idx = np.random.randint(low=0, high=224-self.patch_size+1, size=3)
        self.lr = self.lr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]
        self.hr = self.hr[random_idx[0]:random_idx[0]+self.patch_size, random_idx[1]:random_idx[1]+self.patch_size, random_idx[2]:random_idx[2]+self.patch_size]

        non_zero = np.count_nonzero(self.lr)
        non_zero_proportion = (non_zero/self.total_voxel)

        if self.train:
            if (non_zero_proportion < self.ratio):
                return self.__getitem__(idx)
        else:
            if (non_zero_proportion < self.ratio_eval):
                return self.__getitem__(idx)

        self.lr = self.normalize(self.lr)
        self.hr = self.normalize(self.hr)

        sample_lr = torch.unsqueeze(self.lr, 0)
        sample_hr = torch.unsqueeze(self.hr, 0)

        return sample_hr, sample_lr
