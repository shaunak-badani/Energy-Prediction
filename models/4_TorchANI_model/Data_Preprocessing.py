#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# 
# * This notebook serves the purpose of taking data from npz files and 
# converting it into h5 format, so that in can be loaded dynamically 
# into local memory while running the training model
# 
# * It also saves the configuration to be used further into the training
# 

# In[17]:


import os

import h5py
import os
import numpy as np
import pickle
import torchani


# In[2]:


config = {
    'name' : "Benzene",
    'batches' : [10000, 20000, 30000],
    'testing' : [30000, -1],
    'model_path' : "/scratch/shaunak/models/",
    'datasets_path' : '../../datasets/benzene_old_dft.npz',
}

def save_config():
    f = open('model_config', 'ab')
    pickle.dump(config, f)
    f.close()


# In[3]:


# To reset if want to run notebook again, do not do it if you don't wish 
# to delete already generated h5 file
os.system('rm {}.h5'.format(config['name']))


# In[4]:


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()


# In[5]:


data_path = os.path.join(path, config['datasets_path'])
new_data_file = os.path.join(path, '{}.h5'.format(config['name']))


# In[6]:


molecule = np.load(data_path)
molecule_name = config['name']
batches = config['batches']
names = ["01", "02", "03"]

print("Batches ,", batches)


# In[7]:


n = molecule['E'].shape[0]
config['testing'][-1] = n
print(config)
os.system('rm model_config')
save_config()


# In[8]:


species_map = {
    6 : "C".encode("utf-8"),
    1 : "H".encode("utf-8"),
    8 : "O".encode("utf-8"),
    7 : "N".encode("utf-8"),
}
mult = 627.5094738898777


# In[9]:


species = list(map(lambda x:species_map[x], molecule['z']))


# In[10]:


print("Species : ", species)


# In[11]:


h5_file = h5py.File(new_data_file, 'w')


# In[12]:


if molecule_name not in h5_file:
    b = h5_file.create_group(molecule_name)
else:
    b = h5_file[molecule_name]


# In[13]:


init = 0
for i in range(len(batches)):
    if names[i] not in b:
        sub_group = b.create_group(names[i])
    else:
        sub_group = b[names[i]]
    last = batches[i]
    print(init, last)
    if "coordinates" not in sub_group:
        sub_group.create_dataset("coordinates", data = molecule['R'][init:last])
    if "energies" not in sub_group:
        sub_group.create_dataset("energies", data = molecule['E'][init:last].flatten() / mult) 
    if "species" not in sub_group:
        sub_group.create_dataset("species", data = species)
    init = last


# In[14]:


print("Done with writing to h5 file : ")
for i in names:
    print(h5_file[molecule_name][i]['coordinates'])
    print(h5_file[molecule_name][i]['energies'])
    print(h5_file[molecule_name][i]['species'])
    print(" ")


# In[15]:


print("Closing h5 file ...")
h5_file.close()


# ## Saving the test model

# In[ ]:


init, last = config['testing']


# In[ ]:


# test_data_file = os.path.join(path, 'tmp_testing.h5')
# test_h5_file = h5py.File(test_data_file, 'w')


# In[ ]:




