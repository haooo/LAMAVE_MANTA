#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script
from fastai.vision import *
import pandas as pd
import shutil


# In[3]:

base_dir = Path(os.path.abspath(os.path.dirname(os.path.dirname(sys.argv[0]))))
print(base_dir)
path = base_dir / 'manta'

# In[4]:


test = ImageList.from_folder(path)


# In[5]:
print(os.getcwd())
learn = load_learner(os.path.join(base_dir, 'MANTA_RUN'), test=ImageList.from_folder(path))


# In[6]:


preds, y = learn.get_preds(ds_type=DatasetType.Test)
df=pd.DataFrame(preds,columns=['MANTA','NON_MANTA'])
my_list = list(df[df['MANTA'] > df['NON_MANTA']].index)


# In[37]:


path2 = base_dir / 'result'
os.makedirs(path2)


# In[38]:


for i in my_list:
    shutil.copy(f'{test.items[i]}', path2)
