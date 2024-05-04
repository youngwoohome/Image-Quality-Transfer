import torch
import numpy as np
import yaml
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from model import ResUnet
from new_testData import supervisedIQT_INF, my_collate
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming your base directory is set correctly
base_dir = '/cluster/project0/IQT_Nigeria/skim/HCP_Harry_x8/test_small'
result_dir = '/cluster/project0/IQT_Nigeria/youngwoo/SEResUnet/SEResUnet/x8/project3rd/diffusionGAN/test_result'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load configs
with open('config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

model = ResUnet(1)
pretrained = configs['result_path']+'/best_valid.pth'
model.load_state_dict(torch.load(pretrained, map_location=device))
model = model.to(device)
model.eval()  # Set the model to inference mode

batch_size = 32

# Loop through each subject folder in the test_small directory
for subject_id in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_id, 'T1w')
    if not os.path.isdir(subject_path):  # Skip if not a directory
        continue

    lrfile = os.path.join(subject_path, 'lr_norm.nii.gz')
    hrfile = os.path.join(subject_path, 'T1w_acpc_dc_restore_brain.nii.gz')

    dataset = supervisedIQT_INF(lrfile)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, shuffle=False, drop_last=False)

    lowres = nib.load(lrfile).get_fdata()[8:248, 8:248, 8:248]
    highres = nib.load(hrfile).get_fdata()
    highres = highres[8:248, 8:248, 8:248]
    highres = (highres - highres.min()) / (highres.max() - highres.min())

    pred_ary = torch.zeros(240, 240, 240)

    start = time.time()

    for i, data in enumerate(test_loader):
        if data is not None:
            patch_input, idx = data
            patch_input = patch_input.to(device)
            outputs = model(patch_input).cpu()
            for j in range(patch_input.shape[0]):
                pred_ary[idx[j][0]+8:idx[j][0]+32-8, idx[j][1]+8:idx[j][1]+32-8, idx[j][2]+8:idx[j][2]+32-8] = outputs[j][0, 8:24, 8:24, 8:24]

    end = time.time()
    print(f"Processed {subject_id} in {end-start} seconds.")

    # Save the outputs
    np.save(os.path.join(result_dir, f'{subject_id}_volume_inf.npy'), pred_ary.numpy())
    np.save(os.path.join(result_dir, f'{subject_id}_volume_gt.npy'), highres)
    np.save(os.path.join(result_dir, f'{subject_id}_volume_lr.npy'), lowres)
