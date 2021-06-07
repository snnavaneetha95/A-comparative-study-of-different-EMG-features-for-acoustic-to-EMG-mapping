import numpy as np
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
from keras.layers import LSTM, Dropout, GRU, Convolution1D,  MaxPooling1D, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
#from keras.metrics import Mean as MEAN
#from keras.metrics import std as STD
from keras import backend as K
import sys
import numpy
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
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
#Getting MFCC files

vcvDF = []
vcvDF = pd.DataFrame(vcvDF)
vcvDF.insert(0,"mfcc","Any")
vcvDF.insert(1,"EMG","Any")
mfccTotal = []
emgTotal = []
maximum=-1000

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
  #print(len(y_true),len(y_pred))
  for i in range(1):
     temp=concordance_correlation_coefficient(y_true[:,:,i],y_pred[:,:,i])+temp
  #temp=concordance_correlation_coefficient(y_true,y_pred)
  return (1-temp)/6
#def remove_drift(signal,fs):
#    b,a = signal.butter(5,[3 300],btype='bandpass',fs=600)
#    return signal.filtfilt(b,a,signal)
def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w
def get_emg_features(emg_data):
    #print(emg_data.shape)
    x = emg_data - np.mean(emg_data)
    #x=x/1170
    #x=emg_data
    frame_features = []
    #x = xs[:,i]
    #x=x[:,i]
    w = double_average(x)
    p = x - w
    r = np.abs(p)
    w_h = librosa.util.frame(w, frame_length=20, hop_length=6).mean(axis=0)
    p_w = librosa.feature.rms(w, frame_length=20, hop_length=6, center=False)
    p_w = np.squeeze(p_w, 0)
    p_r = librosa.feature.rms(r, frame_length=20, hop_length=6, center=False)
    p_r = np.squeeze(p_r, 0)
    z_p = librosa.feature.zero_crossing_rate(p, frame_length=20, hop_length=6, center=False)
    z_p = np.squeeze(z_p, 0)
    r_h = librosa.util.frame(r, frame_length=20, hop_length=6).mean(axis=0)
    frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
    #frame_features.append(np.stack([w_h, p_w, p_r, r_h], axis=1))
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

cutoff=1000
fs=16000
order=4
pth='3-300/'

#os.chdir('/home2/data/Manthan/101/')
os.chdir('/home2/data/navaneetha/speech-emg/AudibleUKA/')

for wavFile in sorted(glob.glob("spliced_a_002_101*.wav")):
                    data,samples = librosa.load(wavFile,mono=False)
                    #data, sample = librosa.load(wavFile)
                    data = data[0,:]
                    data = np.asfortranarray(data)
                    meltspec_args = {"n_fft": int(0.032*samples), "hop_length": int(0.010*samples), "window":  get_window("blackman",int(0.032*samples))}
                    # mfccwav = librosa.feature.mfcc(data, samples, n_mfcc= 25, 
                                   # hop_length=int(0.010*samples), 
                                   # n_fft=int(0.025*samples))
                    mfccwav = librosa.feature.mfcc(y=data, sr=samples, S=None, n_mfcc=25, **meltspec_args)
                    #print(mfccwav.shape)
                    #mfccwav=np.array(mfccwav)
                    mfccTotal.append(mfccwav)
                    #print("done MFCC")
                    
                    

print(len(mfccTotal))
#mfccTotal=np.array(mfccTotal)
for i in range(len(mfccTotal)):
    mfccTotal[i] = np.transpose(mfccTotal[i])
    #mfccTotal[i]=(mfccTotal[i]-np.mean(mfccTotal[i],axis=0))/np.std(mfccTotal[i],axis=0)

 

#os.chdir('/home2/data/Manthan/101/')
os.chdir('/home2/data/navaneetha/speech-emg/AudibleUKA/')

emgArr_ch1=[]
emgArr_ch2=[]
emgArr_ch3=[]
emgArr_ch4=[]
emgArr_ch5=[]
emgArr_ch6=[]
for emgFile in sorted(glob.glob("splicedArray_e07_002_101*.mat")):
        emg = si.loadmat(emgFile)
        data=emg['ADC_modified']
        data=data[:6,:]
        data=np.transpose(data)
        #print(emg)
        #print(emg['processed'])
        feat = np.array(get_emg_features(data[:,0]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch1.append(feat)
        feat = np.array(get_emg_features(data[:,1]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch2.append(feat)
        feat = np.array(get_emg_features(data[:,2]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch3.append(feat)
        feat = np.array(get_emg_features(data[:,3]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch4.append(feat)
        feat = np.array(get_emg_features(data[:,4]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch5.append(feat)
        feat = np.array(get_emg_features(data[:,5]))
        #feat=(feat-np.mean(feat,axis=0))/np.std(feat,axis=0)
        emgArr_ch6.append(feat)

print('emg channel1',emgArr_ch1[0].shape)
OutDir = '/home2/data/navaneetha/'
"""
Mfcc=np.asarray(Mfcc)
emgArr_ch1=np.asarray(emgArr_ch1)
emgArr_ch2=np.asarray(emgArr_ch2)
emgArr_ch3=np.asarray(emgArr_ch3)
emgArr_ch4=np.asarray(emgArr_ch4)
emgArr_ch5=np.asarray(emgArr_ch5)
emgArr_ch6=np.asarray(emgArr_ch6)
"""
length=[]  
for i in range(len(emgArr_ch1)):
    #print(i)
    if(len(emgArr_ch1[i]) > len(mfccTotal[i])):
        subLen = len(emgArr_ch1[i]) - len(mfccTotal[i])
        change = len(emgArr_ch1[i]) - subLen
        emgArr_ch1[i] = emgArr_ch1[i][:change,:] 
        emgArr_ch2[i] = emgArr_ch2[i][:change,:]
        emgArr_ch3[i] = emgArr_ch3[i][:change,:]
        emgArr_ch4[i] = emgArr_ch4[i][:change,:]
        emgArr_ch5[i] = emgArr_ch5[i][:change,:]
        emgArr_ch6[i] = emgArr_ch6[i][:change,:]      
    if(len(mfccTotal[i]) > len(emgArr_ch1[i])):
        subLen = len(mfccTotal[i]) - len(emgArr_ch1[i])
        change = len(mfccTotal[i]) - subLen
        #print(change)
        #print(len(mfccTotal[i]))
        mfccTotal[i] = mfccTotal[i][:change][:]
    #print(Mfcc[i].shape,emgArr_ch1[i].shape)
    #mfccTotal[i]=mfccTotal[i][20:len(mfccTotal[i])-20,:]
    #emgTotal[i]=emgTotal[i][20:len(emgTotal[i])-20,:]
    #print(mfccTotal[i].shape,emgTotal[i].shape)
    if i==0:
        m=len(mfccTotal[i])
    if m<len(mfccTotal[i]):
        m=len(mfccTotal[i])
    if len(mfccTotal[i])<=0:
        print('lalalalalalalalalalalalalalalalalalalalala')
        print(len(mfccTotal[i]))
        break
    length.append(len(mfccTotal[i]))

#print(Mfcc[0].shape)
#print(emgArr_ch1[0].shape)
#print(m)
NoUnits=128 #LSTM units
BatchSize=16
NoEpoch=75
std_frac=0.25
n_mfcc=25
inputDim=25#775
outputdim = 6
n_feats=5
kfold = KFold(n_splits=20, shuffle=False)
X1=np.asarray(mfccTotal)
y=np.asarray(emgArr_ch1)

CorrAll_k_ch1=[]
CorrAll_std_k_ch1=[]
CorrAll_k_ch2=[]
CorrAll_std_k_ch2=[]
CorrAll_k_ch3=[]
CorrAll_std_k_ch3=[]
CorrAll_k_ch4=[]
CorrAll_std_k_ch4=[]
CorrAll_k_ch5=[]
CorrAll_std_k_ch5=[]
CorrAll_k_ch6=[]
CorrAll_std_k_ch6=[]
td_1=[]    
td_2=[]
td_3=[]
td_4=[]
td_5=[]
td_1_ccc=[]    
td_2_ccc=[]
td_3_ccc=[]
td_4_ccc=[]
td_5_ccc=[]
count = 0
for train , test in kfold.split(np.asarray(X1),np.asarray(y)):
    print('..compiling model')
    train=np.array(train)
    test=np.array(test)
    print(train)
    print(X1[train].shape,np.array(emgArr_ch1)[train].shape)
    ch1_train=np.asarray(np.asarray(emgArr_ch1)[train])
    ch2_train=np.asarray(np.asarray(emgArr_ch2)[train])
    ch3_train=np.asarray(np.asarray(emgArr_ch3)[train])
    ch4_train=np.asarray(np.asarray(emgArr_ch4)[train])
    ch5_train=np.asarray(np.asarray(emgArr_ch5)[train])
    ch6_train=np.asarray(np.asarray(emgArr_ch6)[train])
    

    mdninput_Lstm= keras.Input(shape=(None,inputDim))
    mdninput_Lstm_1= Masking(mask_value=0.)(mdninput_Lstm)
    lstm_1=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh', dropout = 0.2))(mdninput_Lstm_1)
    lstm_2a=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh', dropout = 0.2))(lstm_1)
    lstm_2=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2a)
    lstm_2b=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2)
    lstm_2c=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2b)
    output_ch1=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    output_ch2=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    output_ch3=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    output_ch4=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    output_ch5=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    output_ch6=TimeDistributed(Dense(n_feats, activation='linear'))(lstm_2c)
    model = keras.models.Model(inputs=mdninput_Lstm,outputs=[output_ch1,output_ch2,output_ch3,output_ch4,output_ch5,output_ch6])
    model.summary()
    #plot_model(model, to_file='multiple_outputs.png')
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    #loss_funcs={'time_distributed_9':CCC_loss, 'time_distributed_10':CCC_loss, 'time_distributed_11':CCC_loss, 'time_distributed_12':CCC_loss, 'time_distributed_13':CCC_loss, 'time_distributed_14':CCC_loss}
    #loss_weights={'time_distributed_9':1, 'time_distributed_10':1, 'time_distributed_11':1, 'time_distributed_12':1, 'time_distributed_13':1, 'time_distributed_14':1}
    opt = keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=opt,loss='mse')
    
    OutFileName='BLSTMforAEIusing5TD_mse'
    fName=OutFileName
  
    print('..fitting model')

    checkpointer = ModelCheckpoint(filepath=OutDir+fName + '_.h5', verbose=0, save_best_only=True)
    checkpointer1 = ModelCheckpoint(filepath=OutDir+fName + '_weights.h5', verbose=0, save_best_only=True)
    earlystopper =EarlyStopping(monitor='val_loss', patience=7)
    print(X1[train].shape,y[train].shape)
    X_train=X1[train]
    #Y_train=y[train]
    #print(X_train[0].shape)
    
    meanStd_1 = []
    meanStd_2 = []
    meanStd_3 = []
    meanStd_4 = []
    meanStd_5 = []
    meanStd_6 = []
    for i in range(0,n_feats):
        x_1 = []
        x_2 = []
        x_3 = []
        x_4 = []
        x_5 = []
        x_6 = []
        for j in range(len(ch1_train)):
            x_1.append(list(ch1_train[j][:,i]))
            x_2.append(list(ch2_train[j][:,i]))
            x_3.append(list(ch3_train[j][:,i]))
            x_4.append(list(ch4_train[j][:,i]))
            x_5.append(list(ch5_train[j][:,i]))
            x_6.append(list(ch6_train[j][:,i]))
        xFinal_1 = sum(x_1,[])
        xFinal_2 = sum(x_2,[])
        xFinal_3 = sum(x_3,[])
        xFinal_4 = sum(x_4,[])
        xFinal_5 = sum(x_5,[])
        xFinal_6 = sum(x_6,[])
        xMean_1 = np.mean(np.array(xFinal_1))
        xMean_2 = np.mean(np.array(xFinal_2))
        xMean_3 = np.mean(np.array(xFinal_3))
        xMean_4 = np.mean(np.array(xFinal_4))
        xMean_5 = np.mean(np.array(xFinal_5))
        xMean_6 = np.mean(np.array(xFinal_6))
        xStd_1 = np.std(np.array(xFinal_1))
        xStd_2 = np.std(np.array(xFinal_2))
        xStd_3 = np.std(np.array(xFinal_3))
        xStd_4 = np.std(np.array(xFinal_4))
        xStd_5 = np.std(np.array(xFinal_5))
        xStd_6 = np.std(np.array(xFinal_6))
        xTotal_1 = [xMean_1, xStd_1]
        xTotal_2 = [xMean_2, xStd_2]
        xTotal_3 = [xMean_3, xStd_3]
        xTotal_4 = [xMean_4, xStd_4]
        xTotal_5 = [xMean_5, xStd_5]
        xTotal_6 = [xMean_6, xStd_6]
        meanStd_1.append(xTotal_1)
        meanStd_2.append(xTotal_2)
        meanStd_3.append(xTotal_3)
        meanStd_4.append(xTotal_4)
        meanStd_5.append(xTotal_5)
        meanStd_6.append(xTotal_6)
        #print(xMean.shape)
        #print(i)    
           
    for i in range(len(ch1_train)):
            for j in range(len(meanStd_1)):
                mean_1 = meanStd_1[j][0]
                mean_2 = meanStd_2[j][0]
                mean_3 = meanStd_3[j][0]
                mean_4 = meanStd_4[j][0]
                mean_5 = meanStd_5[j][0]
                mean_6 = meanStd_6[j][0]
                std_1 = meanStd_1[j][1]
                std_2 = meanStd_2[j][1]
                std_3 = meanStd_3[j][1]
                std_4 = meanStd_4[j][1]
                std_5 = meanStd_5[j][1]
                std_6 = meanStd_6[j][1]
                for k in range(len(ch1_train[i])):
                    #print(np.array(ch1_train).shape)
                    #print(np.array(ch1_train[i]).shape)
                    ch1_train[i][k,j] = ((ch1_train[i][k,j] - mean_1)/std_1)
                    ch2_train[i][k,j] = ((ch2_train[i][k,j] - mean_2)/std_2)
                    ch3_train[i][k,j] = ((ch3_train[i][k,j] - mean_3)/std_3)
                    ch4_train[i][k,j] = ((ch4_train[i][k,j] - mean_4)/std_4)
                    ch5_train[i][k,j] = ((ch5_train[i][k,j] - mean_5)/std_5)
                    ch6_train[i][k,j] = ((ch6_train[i][k,j] - mean_6)/std_6)
            #print(i)
    
    
    
    
    for i in range(len(X_train)):
        X_train[i] = np.transpose(X_train[i])
        ch1_train[i] = np.transpose(ch1_train[i])
        ch2_train[i] = np.transpose(ch2_train[i])
        ch3_train[i] = np.transpose(ch3_train[i])
        ch4_train[i] = np.transpose(ch4_train[i])
        ch5_train[i] = np.transpose(ch5_train[i])
        ch6_train[i] = np.transpose(ch6_train[i])
    
    
    TT_max=560#np.max(TT_Total)
    for i in range(len(X_train)):
        X_train[i] = np.transpose(pad_sequences(np.array(X_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        
    for i in range(len(ch1_train)):
        ch1_train[i] = np.transpose(pad_sequences(np.array(ch1_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        ch2_train[i] = np.transpose(pad_sequences(np.array(ch2_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        ch3_train[i] = np.transpose(pad_sequences(np.array(ch3_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        ch4_train[i] = np.transpose(pad_sequences(np.array(ch4_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        ch5_train[i] = np.transpose(pad_sequences(np.array(ch5_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        ch6_train[i] = np.transpose(pad_sequences(np.array(ch6_train[i]), padding='post',maxlen=TT_max,dtype='float'))
    



    #print(X_train.shape)
    
    X_train_1=[]
    Y_train_1=[]
    ch1_train_1=[]
    ch2_train_1=[]
    ch3_train_1=[]
    ch4_train_1=[]
    ch5_train_1=[]
    ch6_train_1=[]
    X_train=np.asarray(X_train)
    for i in range(len(ch1_train)):
        K_=np.resize(X_train[i],(len(X_train[i]),25))
        X_train_1.append(np.asarray(K_))
        #X_train_1.append(tf.convert_to_tensor(K_))

        K1=np.resize(ch1_train[i],(len(ch1_train[i]),n_feats))
        ch1_train_1.append(np.asarray(K1))
        #ch1_train_1.append(tf.convert_to_tensor(K1))
        #ch1_train_1.append(tf.convert_to_tensor(ch1_train[i]))

        K2=np.resize(ch2_train[i],(len(ch2_train[i]),n_feats))
        ch2_train_1.append(np.asarray(K2))
        #ch2_train_1.append(tf.convert_to_tensor(K2))

        K3=np.resize(ch3_train[i],(len(ch3_train[i]),n_feats))
        ch3_train_1.append(np.asarray(K3))
        #ch3_train_1.append(tf.convert_to_tensor(K3))

        K4=np.resize(ch4_train[i],(len(ch4_train[i]),n_feats))  
        ch4_train_1.append(np.asarray(K4))    
        #ch4_train_1.append(tf.convert_to_tensor(K4))

        K5=np.resize(ch5_train[i],(len(ch5_train[i]),n_feats))
        ch5_train_1.append(np.asarray(K5))
        #ch5_train_1.append(tf.convert_to_tensor(K5))
        
        K6=np.resize(ch6_train[i],(len(ch6_train[i]),n_feats))
        ch6_train_1.append(np.asarray(K6))
        #ch6_train_1.append(tf.convert_to_tensor(K6))
    print(len(X_train_1))
    #X_train_1=np.asarray(X_train_1)
    #ch1_train_1=np.asarray(ch1_train_1)
    print(X_train_1[1].shape)
    #print(X_train_1[0][0])
    print(np.array(ch1_train_1).shape)
    print(ch1_train_1[1].shape)
    #print(np.asarray(X_train_1[0][0]))
    #print(np.asarray(ch1_train_1[0][0]))
    print('model fitttttttttinggggggggg')
    #X_train_1=np.asarray(X_train_1).astype('float32')
    #print(hidden_1.shape)
    #print(output_ch1.shape)
    model = tf.keras.models.load_model(OutDir+fName + '_.h5')
    #history=model.fit(np.asarray(X_train_1),[np.asarray(ch1_train_1),np.asarray(ch2_train_1),np.asarray(ch3_train_1),np.asarray(ch4_train_1),np.asarray(ch5_train_1),np.asarray(ch6_train_1)],validation_split=0.030,epochs=NoEpoch, batch_size=BatchSize,verbose=1,shuffle=True, callbacks=[checkpointer,checkpointer1,earlystopper])
    
    
    yTest_ch1=np.asarray(np.asarray(emgArr_ch1)[test])
    yTest_ch2=np.asarray(np.asarray(emgArr_ch2)[test])
    yTest_ch3=np.asarray(np.asarray(emgArr_ch3)[test])
    yTest_ch4=np.asarray(np.asarray(emgArr_ch4)[test])
    yTest_ch5=np.asarray(np.asarray(emgArr_ch5)[test])
    yTest_ch6=np.asarray(np.asarray(emgArr_ch6)[test])
    X1_test=np.asarray(X1[test])
    for i in range(len(yTest_ch1)):
            for j in range(len(meanStd_1)):
                mean_1 = meanStd_1[j][0]
                mean_2 = meanStd_2[j][0]
                mean_3 = meanStd_3[j][0]
                mean_4 = meanStd_4[j][0]
                mean_5 = meanStd_5[j][0]
                mean_6 = meanStd_6[j][0]
                std_1 = meanStd_1[j][1]
                std_2 = meanStd_2[j][1]
                std_3 = meanStd_3[j][1]
                std_4 = meanStd_4[j][1]
                std_5 = meanStd_5[j][1]
                std_6 = meanStd_6[j][1]
                for k in range(len(X1_test[i])):
                    yTest_ch1[i][k,j] = ((yTest_ch1[i][k,j] - mean_1)/std_1)
                    yTest_ch2[i][k,j] = ((yTest_ch2[i][k,j] - mean_2)/std_2)
                    yTest_ch3[i][k,j] = ((yTest_ch3[i][k,j] - mean_3)/std_3)
                    yTest_ch4[i][k,j] = ((yTest_ch4[i][k,j] - mean_4)/std_4)
                    yTest_ch5[i][k,j] = ((yTest_ch5[i][k,j] - mean_5)/std_5)
                    yTest_ch6[i][k,j] = ((yTest_ch6[i][k,j] - mean_6)/std_6)
            
            #print(j)
    
    
    X1_test_1=[]
    for i in range(len(X1_test)):
        K_=np.transpose(np.transpose(np.asarray(X1_test[i])))
        K_=np.resize(K_,(len(K_),25))
        X1_test_1.append(np.asarray(K_))
        #X_train_1.append(tf.convert_to_tensor(K_))

    #y_test=y[test]
    X_test = []
    #Y_test = []
    for i in np.arange(len(X1_test_1)):
                #E_t = y_test[i]
                M_t = X1_test_1[i]
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
    yPred_ch1=[]
    yPred_ch2=[]
    yPred_ch3=[]
    yPred_ch4=[]
    yPred_ch5=[]
    yPred_ch6=[]
    XtrainMat = []
    
    print(len(X_test))
    for x in range(len(X_test)):
        y_pred = model.predict(np.asarray(X_test[x]))
        yPred_ch1.append(y_pred[0])
        yPred_ch2.append(y_pred[1])
        yPred_ch3.append(y_pred[2])
        yPred_ch4.append(y_pred[3])
        yPred_ch5.append(y_pred[4])
        yPred_ch6.append(y_pred[5])
        #print(np.asarray(y_pred).shape)
        #print(np.asarray(y_pred[0]).shape)
        yPred.append(y_pred)
        #print((np.array(y_pred)[:,0,:,:]).shape)
        XtrainMat.append(np.array((np.array(y_pred)[0,0,:,:])))
        XtrainMat.append(np.array((np.array(y_pred)[1,0,:,:])))
        XtrainMat.append(np.array((np.array(y_pred)[2,0,:,:])))
        XtrainMat.append(np.array((np.array(y_pred)[3,0,:,:])))
        XtrainMat.append(np.array((np.array(y_pred)[4,0,:,:])))
        XtrainMat.append(np.array((np.array(y_pred)[5,0,:,:])))
        #print(np.array((np.array(y_pred)[5,0,:,:])).shape)
    #XtrainMat = np.transpose(XtrainMat,(0,2,1))
    print('*********************',np.array(XtrainMat).shape)
    #savemat('/home2/data/navaneetha/EstimatedfromAEI.mat',{'XtrainMat':XtrainMat})
    
    yTest_ch1=np.asarray(np.asarray(emgArr_ch1)[test])
    yTest_ch2=np.asarray(np.asarray(emgArr_ch2)[test])
    yTest_ch3=np.asarray(np.asarray(emgArr_ch3)[test])
    yTest_ch4=np.asarray(np.asarray(emgArr_ch4)[test])
    yTest_ch5=np.asarray(np.asarray(emgArr_ch5)[test])
    yTest_ch6=np.asarray(np.asarray(emgArr_ch6)[test])
    X1_test=np.asarray(X1[test])
    for i in range(len(yTest_ch1)):
            for j in range(len(meanStd_1)):
                mean_1 = meanStd_1[j][0]
                mean_2 = meanStd_2[j][0]
                mean_3 = meanStd_3[j][0]
                mean_4 = meanStd_4[j][0]
                mean_5 = meanStd_5[j][0]
                mean_6 = meanStd_6[j][0]
                std_1 = meanStd_1[j][1]
                std_2 = meanStd_2[j][1]
                std_3 = meanStd_3[j][1]
                std_4 = meanStd_4[j][1]
                std_5 = meanStd_5[j][1]
                std_6 = meanStd_6[j][1]
                for k in range(len(X1_test[i])):
                    yTest_ch1[i][k,j] = ((yTest_ch1[i][k,j] - mean_1)/std_1)
                    yTest_ch2[i][k,j] = ((yTest_ch2[i][k,j] - mean_2)/std_2)
                    yTest_ch3[i][k,j] = ((yTest_ch3[i][k,j] - mean_3)/std_3)
                    yTest_ch4[i][k,j] = ((yTest_ch4[i][k,j] - mean_4)/std_4)
                    yTest_ch5[i][k,j] = ((yTest_ch5[i][k,j] - mean_5)/std_5)
                    yTest_ch6[i][k,j] = ((yTest_ch6[i][k,j] - mean_6)/std_6)
            
            #print(j)
    
    
    X1_test_1=[]
    for i in range(len(X1_test)):
        K_=np.transpose(np.transpose(np.asarray(X1_test[i])))
        K_=np.resize(K_,(len(K_),25))
        X1_test_1.append(np.asarray(K_))
        #X_train_1.append(tf.convert_to_tensor(K_))

        #y_test=y[test]
    X_test = []
    #Y_test = []
    for i in np.arange(len(X1_test_1)):
                #E_t = y_test[i]
                M_t = X1_test_1[i]
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
    yPred_ch1=[]
    yPred_ch2=[]
    yPred_ch3=[]
    yPred_ch4=[]
    yPred_ch5=[]
    yPred_ch6=[]
    for x in range(len(X_test)):
        y_pred = model.predict(np.asarray(X_test[x]))
        yPred_ch1.append(y_pred[0])
        yPred_ch2.append(y_pred[1])
        yPred_ch3.append(y_pred[2])
        yPred_ch4.append(y_pred[3])
        yPred_ch5.append(y_pred[4])
        yPred_ch6.append(y_pred[5])
        print(np.asarray(y_pred).shape)
        print(np.asarray(y_pred[0]).shape)
        yPred.append(y_pred)
        print("done")


    print(yTest_ch1.shape)
    for i in range(len(yPred)):
        yPred_ch1[i] = np.squeeze(yPred_ch1[i],axis=0)
        yPred_ch2[i] = np.squeeze(yPred_ch2[i],axis=0)
        yPred_ch3[i] = np.squeeze(yPred_ch3[i],axis=0)
        yPred_ch4[i] = np.squeeze(yPred_ch4[i],axis=0)
        yPred_ch5[i] = np.squeeze(yPred_ch5[i],axis=0)
        yPred_ch6[i] = np.squeeze(yPred_ch6[i],axis=0)
        #print(yTest_ch1[i].shape)
        #Y_test[i] = np.squeeze(Y_test[i],axis=0)
    
    
    

    #Corr2
    CorrAll_1 = []
    CorrAll_std_1=[]
    CorrAll_2 = []
    CorrAll_std_2=[]
    CorrAll_3 = []
    CorrAll_std_3=[]
    CorrAll_4 = []
    CorrAll_std_4=[]
    CorrAll_5 = []
    CorrAll_std_5=[]
    #CorrAll_6 = []
    #CorrAll_std_6=[]
    #ErrAll=[]
    

    for i in range(0,n_feats):
        Corr_1 = []
        Corr_2 = []
        Corr_3 = []
        Corr_4 = []
        Corr_5 = []
        #Corr_6 = []
        z=[]
        z_ccc=[]
        for j in range(len(yPred)):
               yPred1 = yPred_ch1[j][:,i]
               yTest1 = yTest_ch1[j][:,i]
               #print(np.array(yPred1).shape)
               c=CCC(yTest1,yPred1)
               z_ccc.append(c)
               Corr_1.append(c)
               c,_ = pearsonr(yTest1,yPred1)
               z.append(c)
               #Corr_1.append(c)
               yPred1 = yPred_ch2[j][:,i]
               yTest1 = yTest_ch2[j][:,i]
               c=CCC(yTest1,yPred1)
               z_ccc.append(c)
               Corr_2.append(c)
               c,_ = pearsonr(yTest1,yPred1)
               z.append(c)
               #Corr_2.append(c)
               yPred1 = yPred_ch3[j][:,i]
               yTest1 = yTest_ch3[j][:,i]
               c=CCC(yTest1,yPred1)
               z_ccc.append(c)
               Corr_3.append(c)
               c,_ = pearsonr(yTest1,yPred1)
               z.append(c)
               #Corr_3.append(c)
               yPred1 = yPred_ch4[j][:,i]
               yTest1 = yTest_ch4[j][:,i]
               c=CCC(yTest1,yPred1)
               z_ccc.append(c)
               Corr_4.append(c)
               c,_ = pearsonr(yTest1,yPred1)
               z.append(c)
               #Corr_4.append(c)
               yPred1 = yPred_ch5[j][:,i]
               yTest1 = yTest_ch5[j][:,i]
               c=CCC(yTest1,yPred1)
               z_ccc.append(c)
               Corr_5.append(c)
               c,_ = pearsonr(yTest1,yPred1)
               z.append(c)
               #Corr_5.append(c)
               #yPred1 = yPred_ch6[j][:,i]
               #yTest1 = yTest_ch6[j][:,i]
               #c=CCC(yTest1,yPred1)
               #z_ccc.append(c)
               #Corr_6.append(c)
               #c,_ = pearsonr(yTest1,yPred1)
               #z.append(c)
               #Corr_6.append(c)
               #c,_ = pearsonr(yTest1,yPred1)
               #print(c)
        if i==0:
               td_1.append(np.mean(z))
               td_1_ccc.append(np.mean(z_ccc))
        elif i==1:
               td_2.append(np.mean(z))
               td_2_ccc.append(np.mean(z_ccc))
        elif i==2:
               td_3.append(np.mean(z))
               td_3_ccc.append(np.mean(z_ccc))
        elif i==3:
               td_4.append(np.mean(z))
               td_4_ccc.append(np.mean(z_ccc))
        elif i==4:
               td_5.append(np.mean(z))
               td_5_ccc.append(np.mean(z_ccc))
        CorMean_1 = (np.mean(np.array(Corr_1),axis=0))
        CorMean_2 = (np.mean(np.array(Corr_2),axis=0))
        CorMean_3 = (np.mean(np.array(Corr_3),axis=0))
        CorMean_4 = (np.mean(np.array(Corr_4),axis=0))
        CorMean_5 = (np.mean(np.array(Corr_5),axis=0))
        #CorMean_6 = (np.mean(np.array(Corr_6),axis=0))
        #ErrAll.append(np.mean(np.array(Err),axis=0))
        CorStd_1=np.std(np.array(Corr_1),axis=0)
        CorStd_2=np.std(np.array(Corr_2),axis=0)
        CorStd_3=np.std(np.array(Corr_3),axis=0)
        CorStd_4=np.std(np.array(Corr_4),axis=0)
        CorStd_5=np.std(np.array(Corr_5),axis=0)
        #CorStd_6=np.std(np.array(Corr_6),axis=0)
        CorrAll_std_1.append(CorStd_1)
        CorrAll_std_2.append(CorStd_2)
        CorrAll_std_3.append(CorStd_3)
        CorrAll_std_4.append(CorStd_4)
        CorrAll_std_5.append(CorStd_5)
        #CorrAll_std_6.append(CorStd_6)
        CorrAll_1.append(CorMean_1)
        CorrAll_2.append(CorMean_2)
        CorrAll_3.append(CorMean_3)
        CorrAll_4.append(CorMean_4)
        CorrAll_5.append(CorMean_5)
        #CorrAll_6.append(CorMean_6)
    
               
    #CorrTotal = (np.mean(np.array(CorrAll),axis=0))
    #Total.append(CorrTotal)
    #print(CorrAll_2)
    #print(CorrAll_std_2)
    #print(np.mean(np.array(CorrAll_2),axis=0))
    CorrAll_k_ch1.append(CorrAll_1)
    CorrAll_std_k_ch1.append(CorrAll_std_1)
    CorrAll_k_ch2.append(CorrAll_2)
    CorrAll_std_k_ch2.append(CorrAll_std_2)
    CorrAll_k_ch3.append(CorrAll_3)
    CorrAll_std_k_ch3.append(CorrAll_std_3)
    CorrAll_k_ch4.append(CorrAll_4)
    CorrAll_std_k_ch4.append(CorrAll_std_4)
    CorrAll_k_ch5.append(CorrAll_5)
    CorrAll_std_k_ch5.append(CorrAll_std_5)
    #CorrAll_k_ch6.append(CorrAll_6)
    #CorrAll_std_k_ch6.append(CorrAll_std_6)
    print(CorrAll_1)
    print(CorrAll_2)
    print(CorrAll_3)
    print(CorrAll_4)
    print(CorrAll_5)
    #print(CorrAll_6)
    if maximum<np.mean([np.mean(CorrAll_1),np.mean(CorrAll_2),np.mean(CorrAll_3),np.mean(CorrAll_4),np.mean(CorrAll_5)]):
        maximum=np.mean([np.mean(CorrAll_1),np.mean(CorrAll_2),np.mean(CorrAll_3),np.mean(CorrAll_4),np.mean(CorrAll_5)])
    count=count+1
    break
    

print(np.mean(CorrAll_k_ch1))
print(np.mean(CorrAll_std_k_ch1))
print(np.mean(CorrAll_k_ch2))
print(np.mean(CorrAll_std_k_ch2))
print(np.mean(CorrAll_k_ch3))
print(np.mean(CorrAll_std_k_ch3))
print(np.mean(CorrAll_k_ch4))
print(np.mean(CorrAll_std_k_ch4))
print(np.mean(CorrAll_k_ch5))
print(np.mean(CorrAll_std_k_ch5))
#print(np.mean(CorrAll_k_ch6))
#print(np.mean(CorrAll_std_k_ch6))
print('CC')
print(np.mean(td_1),np.std(td_1))
print(np.mean(td_2),np.std(td_2))
print(np.mean(td_3),np.std(td_3))
print(np.mean(td_4),np.std(td_4))
print(np.mean(td_5),np.std(td_5))
print('CCC')
print(np.mean(td_1_ccc),np.std(td_1_ccc))
print(np.mean(td_2_ccc),np.std(td_2_ccc))
print(np.mean(td_3_ccc),np.std(td_3_ccc))
print(np.mean(td_4_ccc),np.std(td_4_ccc))
print(np.mean(td_5_ccc),np.std(td_5_ccc))
print('CCC max: ',maximum)

dictMat = {'yPred_ch1': yPred_ch1,'yTest_ch1':yTest_ch1,'yPred_ch2': yPred_ch2,'yTest_ch2':yTest_ch2,'yPred_ch3': yPred_ch3,'yTest_ch3':yTest_ch3,'yPred_ch4': yPred_ch4,'yTest_ch4':yTest_ch4,'yPred_ch5': yPred_ch5,'yTest_ch5':yTest_ch5,'yPred_ch6': yPred_ch6,'yTest_ch6':yTest_ch6}
os.chdir('/home2/data/navaneetha/')
savemat('002PredictedandTestAEI_TD.mat',dictMat)


