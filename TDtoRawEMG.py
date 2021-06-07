# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:17:38 2021

@author: Navaneetha
"""

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import scipy.io as si
import os
import glob
import librosa
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.layers import Masking
from keras.utils.generic_utils import Progbar
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from keras.layers import LSTM, Dropout, GRU, Conv1D,  MaxPooling1D, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
#from keras.metrics import Mean as MEAN
#from keras.metrics import std as STD
from keras import backend as K
import sys
import copy
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy.io import savemat
import scipy.io.wavfile as wf
import scipy.signal
from scipy.signal import butter,filtfilt
from scipy.io import wavfile
import sys
from keras.optimizers import SGD,Adam
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.signal import get_window
from keras.utils import plot_model
from scipy.stats.stats import pearsonr
from scipy import signal
from keras.models import load_model

def CCC(y_true, y_pred):
  cor=np.corrcoef(y_true,y_pred)[0][1]
  #cor=correlation_coefficient_loss(y_true,y_pred)
  mean_true=np.mean(y_true)
  mean_pred=np.mean(y_pred)
  var_true=np.var(y_true)
  var_pred=np.var(y_pred)
  sd_true=np.std(y_true)
  sd_pred=np.std(y_pred)
  numerator=2*cor*sd_true*sd_pred
  denominator=var_true+var_pred+(mean_true-mean_pred)**2
  return numerator/denominator
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    #r = K.maximum(K.minimum(r, 1.0), -1.0)
    return (r)

def CC(y_true, y_pred):
    #normalise
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/bottom
    return K.mean(result)

def concordance_correlation_coefficient(y_true, y_pred):
  #cor=np.corrcoef(y_true,y_pred)[0][1]
  #print(y_true)
  #print(y_pred)
  #print(len(y_true),len(y_pred))
  cor=correlation_coefficient_loss(y_true, y_pred)
  #cor=CC(y_true,y_pred)
  #cor=pearsonr(y_true,y_pred)
  mean_true=K.mean(y_true)
  mean_pred=K.mean(y_pred)
  var_true=K.var(y_true)
  var_pred=K.var(y_pred)
  sd_true=K.std(y_true)
  sd_pred=K.std(y_pred)
  numerator=2*cor*sd_true*sd_pred
  denominator=var_true+var_pred+(mean_true-mean_pred)**2
  return numerator/denominator


def CCC_loss(y_true,y_pred):
  temp=0
  print(len(y_true),len(y_pred))
  for i in range(6):
     temp=concordance_correlation_coefficient(y_true[:,:,i],y_pred[:,:,i])+temp
  #temp=concordance_correlation_coefficient(y_true,y_pred)
  return (6-temp)/6
  

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)
def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w
def get_emg_features(emg_data,f_l,h_l):
    #print(emg_data.shape)
    x = emg_data - np.mean(emg_data)
    #x=emg_data
    frame_features = []
    #x = xs[:,i]
    #x=x[:,i]
    w = double_average(x)
    p = x - w
    r = np.abs(p)
    w_h = librosa.util.frame(w, frame_length=f_l, hop_length=h_l).mean(axis=0)
    p_w = librosa.feature.rms(w, frame_length=f_l, hop_length=h_l, center=False)
    p_w = np.squeeze(p_w, 0)
    p_r = librosa.feature.rms(r, frame_length=f_l, hop_length=h_l, center=False)
    p_r = np.squeeze(p_r, 0)
    #z_p = librosa.feature.zero_crossing_rate(p, frame_length=f_l, hop_length=h_l, center=False)
    #z_p = np.squeeze(z_p, 0)
    r_h = librosa.util.frame(r, frame_length=f_l, hop_length=h_l).mean(axis=0)
    #frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1)) #5TD features
    frame_features.append(np.stack([w_h, p_w, p_r, r_h], axis=1)) #4TD features
    #frame_features.append(np.stack([p_r], axis=1))
    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)
def td2RawEmg(f,outputdim):
    ip= keras.Input(shape=(None,f))
    c1=Conv1D(filters = 128, kernel_size=(5), strides=1, padding="same", activation='tanh')(ip)
    c2=Conv1D(filters = 256, kernel_size=(3), strides=1, padding="same", activation='tanh')(c1)
    c3=Conv1D(filters = 512, kernel_size=(3), strides=1, padding="same", activation='tanh')(c2)
    b1=BatchNormalization()(c3)
    blstm1=Bidirectional(LSTM(256, return_sequences=True,activation='tanh',dropout = 0.4))(b1)
    blstm2=Bidirectional(LSTM(256, return_sequences=True,activation='tanh',dropout = 0.4))(blstm1)
    b2=BatchNormalization()(blstm2)
    op =TimeDistributed(Dense(outputdim, activation='linear'))(b2)
    model = keras.models.Model(inputs=ip,outputs=op)
    return model

Emg_Fs = 600
window_size =int(32*0.001*Emg_Fs)
Hop_len = int(10*0.001*Emg_Fs)
TDTotal = []
emgTotal = []
emg_path = '/home2/data/navaneetha/speech-emg/AudibleUKA/'
OutDir = '/home2/data/navaneetha/speech-emg/EMBC/'
no_emg = 6
maximum=-1000
NoUnits=128 #LSTM units
BatchSize=32
NoEpoch=150
std_frac=0.25
outputdim = Hop_len
window = Hop_len
n_feats=4
TTmax = 500
for emgFile in sorted(glob.glob(emg_path+'*.mat')):
    emg = si.loadmat(emgFile)
    emg = emg['ADC_modified'][:6,:]
    emg = emg.T
    y_framed = []   
    for l in range(0,no_emg,1):
        data = np.reshape(emg[:,l],emg[:,l].shape[0])
        data1 = (data-np.mean(data))/np.std(data)
        #print('data1',data1)
        y_framed=librosa.util.frame(data1,window, Hop_len).astype(np.float64).T # 268x19
        #print(y_framed.shape)
        emgTotal.append(y_framed)
        feat = np.array(get_emg_features(data,window_size, Hop_len))
        savemat('/home2/data/navaneetha/TD_features.mat',{'TD':feat})
        feat = (feat-np.mean(feat,axis = 0))/np.std(feat,axis=0) # 268x5
        print(feat.shape)
        #print('feat',feat)
        TDTotal.append(feat)
    #y_framed = np.transpose(y_framed,(1,2,0))
    #emgTotal.append(y_framed)
for i in range(len(emgTotal)):
    #print(i,len(emgTotal[i]))
    #print(len(TDTotal[i]))
    if(len(emgTotal[i]) > len(TDTotal[i])):
        subLen = len(emgTotal[i]) - len(TDTotal[i])
        change = len(emgTotal[i]) - subLen
        emgTotal[i] = emgTotal[i][:change,:]     
    if(len(TDTotal[i]) > len(emgTotal[i])):
        subLen = len(TDTotal[i]) - len(emgTotal[i])
        change = len(TDTotal[i]) - subLen
        TDTotal[i] = TDTotal[i][:change][:]
    #print(i,len(emgTotal[i]))
    #print(len(TDTotal[i]))

print(np.array(emgTotal).shape)
print(np.array(TDTotal).shape)
print(emgTotal[2].shape)
print(TDTotal[2].shape)


fold_count =0

kfold = KFold(n_splits=12, shuffle=True)
X1=np.asarray(TDTotal)
y=np.asarray(emgTotal)

CC_K = []
CCC_K = []
for train , test in kfold.split(np.asarray(X1),np.asarray(y)):
  train=np.array(train)
  test=np.array(test)
  #print(train)
  print(X1[train].shape,np.array(y)[train].shape)
  print(X1[test].shape)
  X1_train=X1[train]
  Y1_train=y[train]
  X1_test=X1[test]
  Y1_test=y[test]
  X_train = []
  Y_train = []
  td=[]
  td_ccc=[]    

  for i in range(len(X1_train)):
      X_train.append(pad_sequences(np.transpose(X1_train[i]), padding='post',maxlen=TTmax,dtype='float'))
      Y_train.append(pad_sequences(np.transpose(Y1_train[i]), padding='post',maxlen=TTmax,dtype='float'))
      #print(X_train[i].shape)
      #print(Y_train[i].shape)
  X_train = np.transpose(X_train,(0,2,1))
  Y_train = np.transpose(Y_train,(0,2,1))
  #print('after padding',np.array(X_train).shape)
  #print('after padding',np.array(Y_train).shape)
  
  model = td2RawEmg(n_feats,outputdim)
  #model.summary()
  print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
  opt = keras.optimizers.Adam(learning_rate=0.002)
  loss_function = 'CCC_loss'
  print('compiling model')
  model.compile(optimizer=opt,loss=CCC_loss)
  #model.compile(optimizer=opt,loss='mse')
  OutFileName='CNN+BLSTM_5TDtoRawEmg'+'Batch'+ str(BatchSize)+'_Masked_'+str(TTmax)+loss_function
  fName=OutFileName
  print('..fitting model')
  checkpointer = ModelCheckpoint(filepath=OutDir+fName + '_weights.h5', verbose=0, save_best_only=True)
  earlystopper =EarlyStopping(monitor='val_loss', patience=7)
  history=model.fit(X_train,Y_train,validation_split=0.11,epochs=NoEpoch, batch_size=BatchSize,verbose=1,shuffle=True, callbacks=[checkpointer,earlystopper])
  #model = load_weights(OutDir+fName + '_weights.h5')
  X_test = []
  #Y_test = []
  print(len(X1_test))
  for i in np.arange(len(X1_test)):
              #E_t = y_test[i]
              M_t = X1_test[i]
              #W_t=W_t[np.newaxis,:,:,np.newaxis]
              #E_t=E_t[np.newaxis,:,:]
              M_t=M_t[np.newaxis]
              #print(M_t[0].shape)
              #print(M_t[0][0].shape)
              #print(M_t[0][0][0].shape)
              #print(M_t[0][0][0][0].shape)
              #Y_test.append(E_t)
              X_test.append(M_t)
  yPred = []
  for x in range(len(X_test)):
      #print(X_test[x].shape)
      y_pred = model.predict(np.asarray(X_test[x]))
      yPred.append(y_pred)
  print("done predictions ")
  
  for i in range(len(yPred)):
      #print('before squeeze',np.array(yPred[i]).shape)
      yPred[i] = np.squeeze(yPred[i],axis=0)
  
  CorrAll = []
  CorrAll_std=[]
  z=[]
  z_ccc=[]
  for i in range(len(yPred)):
     yPred1 = yPred[i]
     #print('after squeeze',np.array(yPred1).shape)
     #print(np.array(Y1_test[i]).shape)
     yTest1 = Y1_test[i] #np.squeeze(Y1_test[i],axis=0)
     yTest1 = np.array(yTest1).flatten()
     yPred1 = np.array(yPred1).flatten()
     c=CCC(yTest1,yPred1)
     z_ccc.append(c)
     c,_ = pearsonr(yTest1,yPred1)
     #c=np.corrcoefyTest1,yPred1)
     z.append(c)
     td.append(np.mean(z))
     td_ccc.append(np.mean(z_ccc))
  print(np.mean(td))
  print(np.std(td))
  print(np.mean(td_ccc))
  print(np.std(td_ccc))
  CC_K.append(td)
  CCC_K.append(td_ccc)
  fold_count = fold_count +1
  #if(fold_count>6):
      #break#
  break#

dictMat={'CC_K':CC_K,'CCC_K':CCC_K}
savemat(OutDir+'TD2RawEMG_CCandCCC_5TDfeatures_CCC_loss.mat',dictMat)

'''     
#print('CCC max: ',maximum)
dictMat = {'yPred_ch1': yPred_ch1,'yTest_ch1':yTest_ch1,'yPred_ch2': yPred_ch2,'yTest_ch2':yTest_ch2,'yPred_ch3': yPred_ch3,'yTest_ch3':yTest_ch3,'yPred_ch4': yPred_ch4,'yTest_ch4':yTest_ch4,'yPred_ch5': yPred_ch2,'yTest_ch5':yTest_ch5,'yPred_ch6': yPred_ch6,'yTest_ch6':yTest_ch6}
os.chdir('/home2/data/navaneetha/speech-emg/'+pth+'/')
savemat('DNN_TestPred_TD5features_CCC',dictMat)

dictMat={'cor1':CorrAll_k_ch1,'std1':CorrAll_std_k_ch1,'cor2':CorrAll_k_ch2,'std2':CorrAll_std_k_ch2,'cor3':CorrAll_k_ch3,'std3':CorrAll_std_k_ch3,'cor4':CorrAll_k_ch4,'std4':CorrAll_std_k_ch4,'cor5':CorrAll_k_ch5,'std5':CorrAll_std_k_ch5,'cor6':CorrAll_k_ch6,'std6':CorrAll_std_k_ch6,'td_1':td_1,'td_2':td_2,'td_3':td_3,'td_4':td_4,'td_5':td_5,'td_1_ccc':td_1_ccc,'td_2_ccc':td_2_ccc,'td_3_ccc':td_3_ccc,'td_4_ccc':td_4_ccc,'td_5_ccc':td_5_ccc}
savemat('DNN_CCandCCC_TD5features_CCC.mat',dictMat)

'''



'''

f = 5
t = 300


model = td2RawEmg(t,f)
model.summary()
for j in range(np.array(yPred1).shape[0]):
         yPred1 = yPred[i][j]
         #print('after squeeze',np.array(yPred1).shape)
         #print(np.array(Y1_test[i]).shape)
         yTest1 = Y1_test[i][j] #np.squeeze(Y1_test[i],axis=0)
         yTest1 = np.array(yTest1).flatten()
         yPred1 = np.array(yPred1).flattten()
         #print()
         c=CCC(yTest1,yPred1)
         z_ccc.append(c)      
         c,_ = pearsonr(yTest1,yPred1)
         #c=np.corrcoefyTest1,yPred1)
         z.append(c)
         td.append(np.mean(z))
         td_ccc.append(np.mean(z_ccc))
'''