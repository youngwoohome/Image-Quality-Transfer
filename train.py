import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import yaml

from datax8 import CustomDataset
from model3d import ResUnet, Discriminator
import torch.nn.functional as F
###Base model training for GAN and NoGAN
with open('config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

beta_min = 0.1
beta_max = 20
num_timesteps = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_sigma_schedule(num_timesteps, beta_min, beta_max, device):
    n_timestep = num_timesteps
    beta_min = beta_min
    beta_max = beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

sigmas, a_s, _ = get_sigma_schedule(num_timesteps, beta_min, beta_max, device=device)
a_s_cum = np.cumprod(a_s.cpu())
sigmas_cum = np.sqrt(1 - a_s_cum ** 2)

a_s_cum = a_s_cum.to(device)
sigmas_cum = sigmas_cum.to(device)
#x_start is the real data (lr_data)
def q_sample(x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start).to(device)
      
    x_t = extract(a_s_cum, t, x_start.shape) * x_start + \
          extract(sigmas_cum, t, x_start.shape) * noise
    
    return x_t


#########
#x = q_sample()
#t = torch.randint(0, num_timesteps, (real_data.size(0),), device=device)
#netD = discriminator(x,t)
###########

#discriminator train function
def disc_loss(disc, disc_sr, disc_hr, criterion_Disc):
    
    sr_loss = criterion_Disc(disc_sr, torch.zeros_like(disc_sr).to(device))
    hr_loss = criterion_Disc(disc_hr, torch.ones_like(disc_hr).to(device))

    disc_loss = (sr_loss + hr_loss) / 2

    return disc_loss
    

#generator loss function
def gen_loss(sr_slice, hr_slice, adversarial_loss, criterion):  

    #L1 loss
    sup_loss = criterion(sr_slice, hr_slice)

    #Total loss
    gen_loss = 0.9*sup_loss + 0.01*adversarial_loss

    return gen_loss


def train_model(model, disc, n_epochs, num_timesteps, train_loader, val_loader, device):
    os.mkdir(configs['result_path'])

    avg_train_losses = [] # to track the average training loss per epoch as the model trains
    avg_valid_losses = [] # to track the average validation loss per epoch as the model trains

    #initialize loss and optimizer
    criterion = torch.nn.L1Loss()
    criterion_Disc = nn.BCEWithLogitsLoss()


    #hyper parameters
    best = 1e8

    for i in range(0, n_epochs + 1):
        valid_losses = [] 
        train_losses = []

        #########Train#######
        model.train() 
        for _, (hr_data, lr_data) in enumerate(train_loader):
            lr_slice = lr_data.to(device)
            hr_slice = hr_data.to(device)

            if configs['discriminator_train'] == True:
                #train discriminator    
                with torch.no_grad():
                    sr_slice = model(lr_slice)

                disc_optimizer.zero_grad()

                # t = torch.randint(0, 60, (hr_slice.size(0),), device=device)
                # t = 10*t
                t = torch.randint(0, num_timesteps, (hr_slice.size(0),), device=device)

                #x_t
                noisy_pred = q_sample(sr_slice, t)
                noisy_target = q_sample(hr_slice, t)
                
                disc_sr = disc(noisy_pred, t)
                disc_hr = disc(noisy_target, t)

                d_loss = disc_loss(disc, disc_sr, disc_hr, criterion_Disc)
                d_loss.backward()
                disc_optimizer.step()

                #train generator
                model_optimizer.zero_grad()
                sr_slice = model(lr_slice)
                #noisy_pred = q_sample(sr_slice, t)
                # disc_sr = disc(noisy_pred, t)

                # adversarial_loss = criterion_Disc(disc_sr, torch.ones_like(disc_sr).to(device))
                disc_sr = disc(sr_slice, t)
                adversarial_loss = criterion_Disc(disc_sr, torch.ones_like(disc_sr).to(device))

                g_loss = gen_loss(sr_slice, hr_slice, adversarial_loss, criterion)
                g_loss.backward()
                model_optimizer.step()
                
                train_losses.append(g_loss.item())
            else:
                model_optimizer.zero_grad()
                sr_slice = model(lr_slice)
                g_loss = criterion(sr_slice, hr_slice)
                g_loss.backward()
                model_optimizer.step()
                
                train_losses.append(g_loss.item())

        avg_train_losses.append(np.average(np.array(train_losses)))

        #######Validation#########
        if i % 5 == 0:
            model.eval()
            with torch.no_grad():
                np.random.seed(42)
                for _ in range(configs['repeat']):
                    for _, (hr_data, lr_data) in enumerate(val_loader):
                        lr_slice_val = lr_data.to(device)
                        hr_slice_val = hr_data.to(device)

                        sr_slice_val = model(lr_slice_val) # forward pass: compute predicted outputs by passing inputs to the model
                        val_loss = criterion(sr_slice_val, hr_slice_val) # calculate the loss
                        valid_losses.append(val_loss.item()) # record validation loss

                # calculate average loss over an epoch
                avg_valid_loss = np.average(np.array(valid_losses))
                avg_valid_losses.append(avg_valid_loss)

                # load the last checkpoint with the best model
                if avg_valid_loss < best:
                    best = avg_valid_loss
                    torch.save(model.state_dict(), configs['result_path']+'/best_valid.pth')

                pd.DataFrame({'train_loss': avg_train_losses}).to_csv(configs['result_path']+'/train_loss.csv', index=False)
                pd.DataFrame({'valid_loss': avg_valid_losses}).to_csv(configs['result_path']+'/val_loss.csv', index=False)

    return model, avg_train_losses, avg_valid_losses

train_files_hr = glob.glob(configs['train_path']+'/*/*/'+configs['gt_name'])
train_files_lr = glob.glob(configs['train_path']+'/*/*/'+configs['lr_name'])
valid_files_hr = glob.glob(configs['test_path']+'/*/*/'+configs['gt_name'])
valid_files_lr = glob.glob(configs['test_path']+'/*/*/'+configs['lr_name'])
batch_size = configs['batch_size']

train_dataset = CustomDataset(configs, train_files_lr, train_files_hr, train=True)
valid_dataset = CustomDataset(configs, valid_files_lr, valid_files_hr, train=False)

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

#instanize model
model = ResUnet(1).cuda()
disc = Discriminator(1).cuda()

#instanize optimizer
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], eps=1e-04, weight_decay=1e-4)
disc_optimizer= torch.optim.Adam(disc.parameters(), lr=configs['lr'], eps=1e-04, weight_decay=1e-4)

# initialize loss and optimizer
n_epochs = 10000

model, train_loss, val_loss = train_model(model, disc, n_epochs, num_timesteps, train_loader, valid_loader, device)