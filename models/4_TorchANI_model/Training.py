#!/usr/bin/env python
# coding: utf-8

# ## Importing MD17 benzene dataset

# In[1]:


import numpy as np
import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import pickle

import matplotlib
import matplotlib.pyplot as plt
from sys import getsizeof


# In[2]:


def load_pickle():
    f = open('model_config', 'rb')     
    cfg = pickle.load(f)
    f.close()
    return cfg


# In[3]:


cfg = load_pickle()


# In[4]:


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join(path, "{}.h5".format(cfg['name']))


# In[5]:


data = torchani.data.load(dspath)


# ## Training own neural network potential

# In[6]:


# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

# device to run the training
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'


# In[7]:


# Rcr = 5.2000e+00
# Rca = 3.5000e+00
# EtaR = torch.tensor([1.6000000e+01], device=device)
# ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
# Zeta = torch.tensor([3.2000000e+01], device=device)
# ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
# EtaA = torch.tensor([8.0000000e+00], device=device)
# ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00], device=device)
# species_order = ['H', 'C', 'N', 'O']
# num_species = len(species_order)
# aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
# energy_shifter = torchani.utils.EnergyShifter(None)


# In[8]:


# try:
#     path = os.path.dirname(os.path.realpath(__file__))
# except NameError:
#     path = os.getcwd()
# dspath = os.path.join(path, 'download/dataset/ani1-up_to_gdb4/ani_gdb_s01.h5')
batch_size = 1250


# In[9]:


# aev_dim = aev_computer.aev_length


# In[10]:


from Model_Hyperparameters import get_aev_params, get_complete_network
energy_shifter, aev_computer = get_aev_params(device)
H_network,C_network, N_network, O_network, model, nn = get_complete_network(aev_computer, device, return_networks = True)
species_order = ['H', 'C', 'N', 'O']


# In[11]:


training, validation = data.subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order).shuffle().split(0.8, None)
training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)


# In[ ]:


# aev_dim = aev_computer.aev_length

# H_network = torch.nn.Sequential(
#     torch.nn.Linear(aev_dim, 160),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(160, 128),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(128, 96),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(96, 1)
# )

# C_network = torch.nn.Sequential(
#     torch.nn.Linear(aev_dim, 144),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(144, 112),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(112, 96),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(96, 1)
# )

# N_network = torch.nn.Sequential(
#     torch.nn.Linear(aev_dim, 128),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(128, 112),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(112, 96),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(96, 1)
# )

# O_network = torch.nn.Sequential(
#     torch.nn.Linear(aev_dim, 128),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(128, 112),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(112, 96),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(96, 1)
# )

# nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
# print(nn)


# In[ ]:





# In[12]:


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)


# In[13]:


# model = torchani.nn.Sequential(aev_computer, nn).to(device)


# In[14]:


print(nn)


# In[15]:


AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)


# In[15]:


AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)


# In[16]:


latest_checkpoint = 'latest.pt'


# In[17]:


if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])


# In[18]:


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))


# In[19]:


tensorboard = torch.utils.tensorboard.SummaryWriter()


# In[20]:


mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 5
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)


# In[21]:


g = next(iter(training))


# In[25]:


species = g['species']
coordinates = g['coordinates'].float()
energies = g['energies'].float()


# In[26]:


print(coordinates.dtype)


# In[27]:


_, predicted_energies = model((species, coordinates))


# In[28]:


print(species.shape)
print(coordinates.shape)


# In[32]:


true_energies = g['energies'].float()


# In[36]:


print(true_energies.shape)


# In[37]:


print(predicted_energies.shape)


# In[38]:


y_hat = true_energies.cpu().detach().numpy()
y = predicted_energies.cpu().detach().numpy()

