import numpy as np
import htkmfc as htk
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
epoch=10 #Number of iterations to be run on the model while training
trainfile='/home/siddharthm/scd/combined/400-mfcc-labels-gender-train.htk'
#testfile='/home/siddharthm/scd/combined/gamma-labels-gender-test.htk'
valfile='/home/siddharthm/scd/combined/400-mfcc-labels-gender-val.htk'
#Some parameters for training the model
epoch=20 #Number of iterations to be run on the model while training
batch=1024*2 #Batch size to be used while training
direc="/home/siddharthm/scd/scores/"
common_save='mfcc-cnn'
name_val=common_save+'-val'
#name_test=common_save+'-test'

#Now the data has the format that last column has the label, and the rest of stuff needs to be reshaped.
#The format for reshaping is as follows: Rows = Number of filters X Context size(40 in this case)
def cnn_reshaper(Data):
        dat=np.reshape(Data,(Data.shape[0],1,39,40)) #The format is: Number of samples, Channels, Rows, Columns
        return dat

def load_data_train(trainfile):
        print "Getting the training data"
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        data=train_data
        x_train=cnn_reshaper(data[:,:-2]) #Set to different column based on different model
        Y_train=data[:,-2]
        print Y_train.shape
        # print np.where(Y_train==2)
        Y_train=Y_train.reshape(Y_train.shape[0],1)
        y_train=np_utils.to_categorical(Y_train,2)
        gender_train=data[:,-1]
        del data
        return x_train,y_train,gender_train
def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        x_test=cnn_reshaper(data[:,:-2])
        Y_test=data[:,-2]
        print np.where(Y_test==2)
        # Y_test=np.reshape(Y_test,(Y_test.shape[0],1))
        # y_test=np_utils.to_categorical(Y_test,2)
        gender_labels=data[:,-1]
        del data
        return x_test,Y_test,gender_labels
def load_data_val(valfile):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        x_val=cnn_reshaper(data[:,:-2])
        Y_val=data[:,-2]
        Y_val=np.reshape(Y_val,(Y_val.shape[0],1))
        y_val=np_utils.to_categorical(Y_val,2)
        gender_val=data[:,-1]
        del data
        return x_val,y_val,gender_val
def data_saver(data):
        os.chdir('/home/siddharthm/combined')
        f=open(common_save+'-complete.txt','rb+')
        f.write('\n')
        f.write(data)
        f.close()

def metrics(y_val,classes,gender_val):
        #We have to modify this to include metrics to capture variations between male and female and blah-blah
        #initializing the two matrixes to be saved.
        cd_correct_matrix=np.zeros((2,2))
        single_correct_matrix=np.zeros((2,2))
        cd_incorrect_matrix=np.zeros((2,2))
        single_incorrect_matrix=np.zeros((2,2))
        print classes.shape
        #print np.where(classes==1) #classes must be one dimensional vector here

        #We need a matrix, one of correctly classified changes, and the other of incorrectly classified changes.
        for i in range(len(y_val)):
                id1=int(gender_val[i][0])
                id2=int(gender_val[i][1])
                if y_test[i]==1:
                        if classes[i]==1:
                                cd_correct_matrix[id1][id2]+=1
                        else:
                                cd_incorrect_matrix[id1][id2]+=1
                else:
                        if classes[i]==0:
                                single_correct_matrix[id1][id2]+=1
                        else:
                                single_incorrect_matrix[id1][id2]+=1
        data_saver('Speaker changes Correct detected')
        data_saver(cd_correct_matrix)
        data_saver('Speaker changes wrongly classified')
        data_saver(cd_incorrect_matrix)
        data_saver('Single speaker frames correct')
        data_saver(single_correct_matrix)
        data_saver('Single speaker frames wrongly classified')
        data_saver(single_incorrect_matrix)
        ### ------------- ###

#Non-function section
x_train,y_train,gender_train=load_data_train(trainfile)
print "Loading training data complete"
#x_test,y_test,gender_labels=load_data_test(testfile)
#print "Loading testing data complete"
x_val,y_val,gender_val=load_data_val(valfile)
print "Loading validation data complete"
## SHAPE TESTS ###
print "Train Shape: ",x_train.shape," ",y_train.shape
#print "Test Shape: ",x_test.shape," ",y_test.shape
print "Val Shape: ",x_val.shape," ",y_val.shape
###

### THE MODEL and ALL ###
def seq(x_train,y_train,x_val,y_val,x_test,y_test):
        #Defining the structure of the neural network
        #Creating a Network, with 2 Convolutional layers
        model=Sequential()
        model.add(Conv2D(64,(3,3),activation='relu',input_shape=(1,39,40)))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(256,activation='relu')) #Fully connected layer 1
        model.add(Dropout(0.5))
        model.add(Dense(2,activation='softmax')) #Output Layer
        model.summary()
        data_saver(model.summary())
        #Compilation region: Define optimizer, cost function, and the metric?
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        #Fitting region:Get to fit the model, with training data
        checkpointer=ModelCheckpoint(filepath=direc+common_save+'.json',monitor='val_acc',save_best_only=True,save_weights_only=True)

        #Doing the training[fitting]
        model.fit(x_train,y_train,epochs=10,batch_size=batch,validation_data=(x_val,y_val),callbacks=[checkpointer])
        model.save_weights(direc+common_save+'-weights'+'.json') #Saving the weights from the model
        model.save(direc+common_save+'-model'+'.json')#Saving the model as is in its state

        ### SAVING THE VALIDATION DATA ###
        scores=model.predict(x_val,batch_size=batch)
        sio.savemat(direc+name_val+'.mat',{'scores':scores,'ytest':y_val}) #These are the validation scores.
        classes=model.predict_classes(x_val,batch_size=batch)
        ### ------------- ###

        ### SAVING THE TESTING DATA ###
        #scores_test=model.predict(x_test,batch_size=batch)
        #sio.savemat(direc+name_test+'.mat',{'scores':scores_test,'ytest':y_test})
        ### ------------- ###
        # print model.evaluate(x_test,y_test,batch_size=batch)

        #predictions=model.predict(x_val,batch_size=batch)
        #print "Shape of predictions: ", predictions.shape
        #print "Shape of y_test: ",y_test.shape
        return classes

#Non-function section

#y_test,predictions,classes=seq(x_train,y_train,x_val,y_val,x_test,y_test) #Calling the seq model, with 2 hidden layers
classes=seq(x_train,y_train,x_val,y_val,0,0) #Calling the seq model, with 2 hidden layers
# metrics(y_val,classes,gender_val)

