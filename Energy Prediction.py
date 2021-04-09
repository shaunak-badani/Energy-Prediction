#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[69]:

import sys
name = sys.argv[1]


# In[70]:


benzene_data = np.load("{}.npz".format(name))


# In[71]:


print(benzene_data.files)


# In[72]:


for i in benzene_data.files:
    print(i, benzene_data[i].shape)


# In[73]:


configs = benzene_data['R']


# In[74]:


fig = plt.figure(figsize = (15, 10))

for num in range(6):
    ax = fig.add_subplot(2, 3, num + 1, projection = '3d')
    i = configs[num]
    ax.scatter(i[:, 0], i[:, 1], i[:, 2])
    ax.set_title("Energy : {}".format(benzene_data['E'][num][0]))
plt.savefig('configurations.png')


# ### Label statistics

# In[75]:


benzene_data['E'].shape


# In[76]:


hist, a = np.histogram(benzene_data['E'], bins = 10)


# In[77]:


print(a)


# In[78]:


print(hist)


# In[82]:


from matplotlib import colors


# In[80]:


fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7), 
                        tight_layout = True)
for s in ['top', 'bottom', 'left', 'right']: 
    axs.spines[s].set_visible(False) 
    
axs.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.6) 

N, bins, patches = axs.hist(benzene_data['E'], bins = 30)
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())
axs.xaxis.set_ticks_position('none') 
axs.yaxis.set_ticks_position('none')

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
    
axs.set_xlabel("Energy Values", fontsize = 20)
axs.set_ylabel("Frequency", fontsize = 20)

axs.set_title("Benzene Label Statistics", fontsize = 30)

plt.savefig("{}_label_statistics.png".format(name))

