import torch
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from model import ResUnet
from testData import supervisedIQT_INF, my_collate
import nibabel as nib
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cube(data):

    hyp_norm = data

    if len(hyp_norm.shape)>3:
        hyp_norm = hyp_norm[:,:, 2:258, 27:283]
    else:
        hyp_norm = hyp_norm[2:258, 27:283]

    return hyp_norm

test_file = 118528
lrfile =  f'/cluster/project0/IQT_Nigeria/youngwoo/SEResUnet/SEResUnet/x8/project3rd/{test_file}/T1w/lr_norm.nii.gz'
hrfile = f'/cluster/project0/IQT_Nigeria/youngwoo/SEResUnet/SEResUnet/x8/project3rd/{test_file}/T1w/T1w_acpc_dc_restore_brain.nii.gz'
batch_size = 32

dataset = supervisedIQT_INF(lrfile)
test_loader =  DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, shuffle=False, drop_last=False)

# # Open the file and load the file
# with open(os.getcwd()+'/configs/seresunet.yaml') as f:
#     configs = yaml.load(f, Loader=SafeLoader)
with open('config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

#model = SEResUnet(configs)
model = ResUnet(1)
pretrained = configs['result_path']+'/best_valid.pth'

model.load_state_dict(torch.load(pretrained))
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

params = sum([np.prod(p.size()) for p in model.parameters()])
print("Number of params: ", params)

lowres = nib.load(lrfile).get_fdata()
highres = nib.load(hrfile).get_fdata()

#lowres = cube(lowres)
#highres = cube(highres)
lowres = lowres[8:248, 8:248, 8:248]
highres = highres[8:248, 8:248, 8:248]
highres = (highres - highres.min())/(highres.max() - highres.min())
print(lowres.shape, highres.shape)
print(f'lwores: {lowres.shape} highres: {highres.shape}')

lowres_torch = torch.unsqueeze(torch.unsqueeze(torch.tensor(lowres),0),0)
highres_torch = torch.unsqueeze(torch.unsqueeze(torch.tensor(highres),0),0)
min_val = lowres_torch.min()

mean_lr = configs['mean']
std_lr = configs['std']
pred_ary = torch.zeros(240,240,240)
pred_ary = (pred_ary-mean_lr)/std_lr
patch_size = 32
total_voxel = patch_size*patch_size*patch_size
print("Start inferencing!")

start = time.time()

for i,data in enumerate(test_loader):
    if data is not None:
        patch_input, idx = data
        patch_input = patch_input.to(device)
        outputs = model(patch_input)
        outputs = outputs.cpu()

        for j in range(patch_input.shape[0]):
            print("patch_size :",outputs[j].shape)
            print(idx[j])
            pred_ary[idx[j][0]+8:idx[j][0]+32-8,idx[j][1]+8:idx[j][1]+32-8,idx[j][2]+8:idx[j][2]+32-8] = outputs[j][0,8:24,8:24,8:24]

#pred_ary[np.where(lowres==min_val)] = min_val
end = time.time()
print("TIME: {}".format(end-start))
np.save(f'volume{test_file}_ds10_inf.npy', pred_ary.numpy())
#np.save(f'volume{test_file}_gt.npy', highres)
#np.save('volume_lr.npy', lowres)

