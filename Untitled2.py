#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, datetime
import warnings
import numpy as np
import logging
import random
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
tf.keras.callbacks
from sklearn.metrics import mean_squared_error
from keras import callbacks
from keras import layers
from keras import models
from keras.layers import Dropout
import keras
from keras import optimizers
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[2]:


seed = 42
#####
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
#####
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
#####
np.random.seed(seed)
random.seed(seed)


# In[3]:


# Import tensorflow
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
print(tf.__version__)


# In[6]:


dataset = np.load("/Users/mohamedshoala/Documents/third semester/neural networks/project/public_data.npz", allow_pickle=True)


# In[7]:


# Watch keys available in the numpy array
keys = list(dataset.keys())
print('keys in our dataset are: ', keys)
print('*'*100)

# look at data
data = dataset["data"]
no_images = data.shape[0]
size_images = data.shape[1:3]
print('Data shape: ',data.shape)
print('*'*100)

# Lool aat labels
labels = dataset["labels"]
no_labels = labels.shape[0]
print("Labels are : ",labels)
print('*'*100)

# Look at bing balanced or imblanced
_, counts = np.unique(labels,return_counts=True)
no_healthy_images = counts[0]
no_unhealthy_images = counts[1]
info_table_dict = {"no_images":no_images, "image_width": size_images[0],"image_length": size_images[1], "no_labels":no_labels,
                   "no_healthy_images":no_healthy_images,"percentage%":no_healthy_images*100/no_labels,
                   "no_unhealthy_images":no_unhealthy_images,"percentage %":no_unhealthy_images*100/no_labels }
info_table = pd.DataFrame(info_table_dict, index =['value'])
info_table


# In[ ]:




