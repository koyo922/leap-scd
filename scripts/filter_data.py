
# coding: utf-8

# In[7]:


import numpy as np
import htkmfc as htk
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D,
from keras.layers.pooling import MaxPooling2D

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
epoch=10 #Number of iterations to be run on the model while training
trainfile='/home/siddharthm/scd/lists/rawtrainfiles.list'
testfile='/home/siddharthm/scd/lists/rawtestfiles.list'
valfile='/home/siddharthm/scd/lists/rawvalfiles.list'

def changedir():
        os.chdir('/home/siddharthm/scd/combined')
        print "Current working directory: ",os.getcwd()

def filter_data(x):
        ### Filter the data. That is only keep 0 or 1 classes.
        return x[ (x[:,-1]==0)|(x[:,-1]==1)]
def load_data_train(trainfile):
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        data=filter_data(train_data)
        print "Filtered train data shape: ",data.shape
        changedir()
        writer=htk.open(filename_train+'.htk',mode='w',veclen=data.shape[1])
        del data

def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        data=filter_data(data)
        print "Filtered test data shape: ",data.shape
        changedir()
        writer=htk.open(filename_test+'.htk',mode='w',veclen=data.shape[1])
        del data

def load_data_val(valfile):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        data=filter_data(data)
        del data

### SHAPE TESTS ###
print "Train Shape: ",x_train.shape," ",y_train.shape
print "Test Shape: ",x_test.shape," ",y_test.shape
print "Val Shape: ",x_val.shape," ",y_val.shape
###

name_val=common_save+'-val'
name_test=common_save+'-test'

