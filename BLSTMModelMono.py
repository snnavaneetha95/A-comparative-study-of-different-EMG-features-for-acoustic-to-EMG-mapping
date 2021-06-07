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
import sys
import numpy
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
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
#Getting MFCC files

vcvDF = []
vcvDF = pd.DataFrame(vcvDF)
vcvDF.insert(0,"mfcc","Any")
vcvDF.insert(1,"EMG","Any")
mfccTotal = []
emgTotal = []

def concordance_correlation_coefficient(y_true, y_pred):
  cor=np.corrcoef(y_true,y_pred)[0][1]
  mean_true=np.mean(y_true)
  mean_pred=np.mean(y_pred)
  var_true=np.var(y_true)
  var_pred=np.var(y_pred)
  sd_true=np.std(y_true)
  sd_pred=np.std(y_pred)
  numerator=2*cor*sd_true*sd_pred
  denominator=var_true+var_pred+(mean_true-mean_pred)**2
  return numerator/denominator

def cheby_highpass_filter(data,cutoff,fs,order):
  nyq=0.5*fs
  normal_cutoff=cutoff/nyq
  b,a=scipy.signal.cheby1(order,1,normal_cutoff, btype='high', analog=False)
  y=filtfilt(b,a,data)
  return y
  
def cheby_lowpass_filter(data,cutoff,fs,order):
  nyq=0.5*fs
  normal_cutoff=cutoff/nyq
  b,a=scipy.signal.cheby1(order,1,normal_cutoff, btype='low', analog=False)
  y=filtfilt(b,a,data)
  return y
  
cutoff=1000
fs=16000
order=4
pth='3-300/'
#os.chdir('/home2/data/Manthan/101/splicedwav/')
os.chdir('/home2/data/navaneetha/speech-emg/AudibleUKA/')

for wavFile in sorted(glob.glob("spliced_a_008_101*.wav")):
                    data,samples = librosa.load(wavFile,mono=False)
                    #data, sample = librosa.load(wavFile)
                    data = data[0,:]
                    data = np.asfortranarray(data)
                    #noise=np.random.normal(0,1,len(data))
                    #data=data+noise
                    #print(data.shape)
                    #data=cheby_lowpass_filter(data,cutoff,fs,order)
                    #data=cheby_highpass_filter(data,cutoff,fs,order)
                    mfccwav = librosa.feature.mfcc(data, samples, n_mfcc= 25, 
                                   hop_length=int(0.010*samples), 
                                   n_fft=int(0.025*samples))
                    mfccTotal.append(mfccwav)
                    #print("done MFCC")


#os.chdir('/home2/data/Manthan/101/Processedprasanta3-300/')
#os.chdir('/home2/data/navaneetha/speech-emg/DAV/UKA/')
os.chdir('/home2/data/navaneetha/speech-emg/Processeddata/ProcessedDavella/008_001/')

for emgFile in sorted(glob.glob("*.mat")):
        emg = si.loadmat(emgFile)
        emgArr = emg['data']
        #emgArr=emgArr[:,[0,2,3,4,6]]
        #print(emgArr.shape)
        emgTotal.append(emgArr)
        #print("done EMG")

#emgTotal=np.array(emgTotal)
#print(emgTotal.shape)

for i in range(len(mfccTotal)):
    mfccTotal[i] = np.transpose(mfccTotal[i])

length=[]  
for i in range(len(emgTotal)):
    #print(len(emgTotal[i]))
    #print(len(mfccTotal[i]))
    #print(i)
    if(len(emgTotal[i]) > len(mfccTotal[i])):
        subLen = len(emgTotal[i]) - len(mfccTotal[i])
        change = len(emgTotal[i]) - subLen
        emgTotal[i] = emgTotal[i][:change,:]       
    if(len(mfccTotal[i]) > len(emgTotal[i])):
        subLen = len(mfccTotal[i]) - len(emgTotal[i])
        change = len(mfccTotal[i]) - subLen
        mfccTotal[i] = mfccTotal[i][:change,:]
    #print(mfccTotal[i].shape,emgTotal[i].shape)
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


print(np.mean(length))
print(np.std(length))
vcvDF['mfcc'] = mfccTotal
vcvDF['EMG'] = emgTotal     
print(m)
emgCalc = copy.deepcopy(emgTotal)    
    
X1 =mfccTotal
y = emgCalc  
"""
from sklearn.model_selection import train_test_split
XP,X1_test, YP, y_test = train_test_split(X1, y, test_size = 0.10, random_state=42, shuffle=True)
X_train,X_val, Y_train, Y_val = train_test_split(XP, YP, test_size = 0.10, random_state=42, shuffle=True)
"""
#Getting mean and std



#for i in range(len((X1_test))):
#    X1_test[i] = ((pad_sequences(X1_test[i], padding='post',maxlen=TT_max,dtype='float')))

#for i in range(len((y_test))):
#    y_test[i] =((pad_sequences(y_test[i], padding='post',maxlen=TT_max,dtype='float')))

#for j in range(len(y_test)):
#    print(j)
#    y_test[j] =  np.transpose(pad_sequences(y_test[j], padding='post',maxlen=TT_max,dtype='float'))
#for i in range(len((X1_test))):
#    X1_test[i] = np.transpose(X1_test[i]) 
 
OutDir = '/home2/data/navaneetha/speech-emg/EMBC/'
 

NoUnits=128 #LSTM units
BatchSize=16
NoEpoch=150
std_frac=0.25
n_mfcc=25
inputDim=n_mfcc
kfold = KFold(n_splits=20, shuffle=False)
X1=np.array(X1)
y=np.array(y)
#print(X1.shape,y.shape)
CorrAll_k=[]
CorrAll_std_k=[]
Total=[]
for train , test in kfold.split(np.asarray(X1),np.asarray(y)):
    print('..compiling model')
    train=np.array(train)
    print(train)
    print(X1[train].shape,y[train].shape)
    #mdninput_Lstm = keras.Input(shape=(None,inputDim))
    #mdninput_LstmD=TimeDistributed(Dense(200))(mdninput_Lstm)
    
    mdninput_Lstm= keras.Input(shape=(None,inputDim))
    mdninput_Lstm_1= Masking(mask_value=0.)(mdninput_Lstm)
    lstm_1=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh', dropout = 0.2))(mdninput_Lstm_1)
    lstm_2a=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh', dropout = 0.2))(lstm_1)
    lstm_2=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2a)
    lstm_2b=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2)
    lstm_2c=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2b)
    #lstm_2d=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh',dropout = 0.2))(lstm_2c)
    output=TimeDistributed(Dense(6, activation='linear'))(lstm_2c)
    model = keras.models.Model(mdninput_Lstm,output)
    
    """
    model=Sequential()
    model.add(Masking(mask_value=0.,input_shape=(None,inputDim)))
    model.add(Bidirectional(LSTM(NoUnits, return_sequences=True,input_shape=(None,inputDim),activation='tanh')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(6, activation='linear'))) 
    """
    model.summary()
    #print(ll)
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    #adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = SGD(lr=0.01, decay=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='mse')
    #OutFileName='final_mfcc_'+Sub+'_F1_'+str(CNN1_filters)+'_len'+str(CNN1_flength)+'_F2_'+str(CNN2_filters)+'_len'+str(CNN2_flength)+'LSTMunits_'+str(NoUnits)+'_'
    OutFileName='BLSTM'+'Batch'+ str(BatchSize)+'_'+'_LSTMunits_'+str(NoUnits)+'Masked_560_CCC'
    fName=OutFileName
    
    print('..fitting model')
    

    checkpointer = ModelCheckpoint(filepath=OutDir+fName + '_.h5', verbose=0, save_best_only=True)
    checkpointer1 = ModelCheckpoint(filepath=OutDir+fName + '_weights.h5', verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper =EarlyStopping(monitor='val_loss', patience=7)
    print(X1[train].shape,y[train].shape)
    X_train=X1[train]
    Y_train=y[train]
    X1_test=X1[test]
    print(X_train[0].shape)
    
    meanStd = []
    for i in range(0,25):
        x = []
        for j in range(len(X_train)):
            x.append(list(X_train[j][:,i]))
        xFinal = sum(x,[])
        xMean = np.mean(np.array(xFinal))
        xStd = np.std(np.array(xFinal))
        xTotal = [xMean, xStd]
        meanStd.append(xTotal)
        #print(i)    
           
    for i in range(len(X_train)):
            for j in range(len(meanStd)):
                mean = meanStd[j][0]
                std = meanStd[j][1]
                for k in range(len(X_train[i])):
                    X_train[i][k,j] = ((X_train[i][k,j] - mean)/std)
            print(i)
    for i in range(len(X1_test)):
            for j in range(len(meanStd)):
                mean = meanStd[j][0]
                std = meanStd[j][1]
                for k in range(len(X1_test[i])):
                    X1_test[i][k,j] = ((X1_test[i][k,j] - mean)/std)
            print(i)
 

    for i in range(len(X_train)):
        X_train[i] = np.transpose(X_train[i])
        Y_train[i] = np.transpose(Y_train[i])
        
    TT_max=560#np.max(TT_Total)
    #padded = pad_sequences(sequences, padding='post',maxlen=TT_max)


    for i in range(len(X_train)):
        X_train[i] = np.transpose(pad_sequences(np.array(X_train[i]), padding='post',maxlen=TT_max,dtype='float'))
        
    for i in range(len(Y_train)):
        Y_train[i] = np.transpose(pad_sequences(np.array(Y_train[i]), padding='post',maxlen=TT_max,dtype='float'))

    print(X_train.shape)
    #X_train=X_train.astype('float32')
    #Y_train=Y_train.astype('float32')  
    print(X_train[0].shape,Y_train[0].shape)
    print(X_train.shape,Y_train.shape)
    X_train_1=[]
    Y_train_1=[]
    for i in range(len(X_train)):
        X_train_1.append(np.asarray(X_train[i]))
        Y_train_1.append(np.asarray(Y_train[i]))
    history=model.fit(np.asarray(X_train_1),np.asarray(Y_train_1),validation_split=0.030,epochs=NoEpoch, batch_size=60,verbose=1,shuffle=True,callbacks=[checkpointer,checkpointer1,earlystopper])

    #model = load_model(OutDir+'BLSTMBatch16__LSTMunits_128__.h5')
    y_test=y[test]
    X_test = []
    Y_test = []
    for i in np.arange(len(X1_test)):
                E_t = y_test[i]
                M_t = X1_test[i]
                #W_t=W_t[np.newaxis,:,:,np.newaxis]
                E_t=E_t[np.newaxis,:,:]
                M_t=M_t[np.newaxis,:,:]
                Y_test.append(E_t)
                X_test.append(M_t)

    yPred = []
    for x in range(len(X_test)):
        y_pred = model.predict(X_test[x])
        yPred.append(y_pred)
        #print("done")


    for i in range(len(yPred)):
        yPred[i] = np.squeeze(yPred[i],axis=0)
        Y_test[i] = np.squeeze(Y_test[i],axis=0) 
        
    from scipy.stats.stats import pearsonr
    """
    #Corr1
    CorrAll = []
    for i in range(len(yPred)):
        yTest1 = Y_test[i]
        yPred1 = yPred[i]
        Cor = []
        for j in range(0,6):
            yTest11 = yTest1[:,j]
            yPred11 = yPred1[:,j]
            c,_ = pearsonr(yTest11,yPred11)
            Cor.append(c)
        CorMean = (np.mean(np.array(Cor),axis=0))
        CorrAll.append(CorMean)

    CorrTotal = (np.mean(np.array(CorrAll),axis=0))
    print(CorrTotal)
    """
    #Corr2
    CorrAll = []
    CorrAll_std=[]
    #ErrAll=[]
    for i in range(0,6):
        Corr = []
        #Err=[]
        for j in range(len(yPred)):
               yPred1 = yPred[j][:,i]
               yTest1 = Y_test[j][:,i]
               #c=concordance_correlation_coefficient(yTest1,yPred1)
               #print(c)
               c,_ = pearsonr(yTest1,yPred1)
               #print(c)
               Corr.append(c)
               #Err.append(mse(np.array(yTest1),np.array(yPred1)))
        CorMean = (np.mean(np.array(Corr),axis=0))
        #ErrAll.append(np.mean(np.array(Err),axis=0))
        CorStd=np.std(np.array(Corr),axis=0)
        CorrAll_std.append(CorStd)
        CorrAll.append(CorMean)
               
    CorrTotal = (np.mean(np.array(CorrAll),axis=0))
    Total.append(CorrTotal)
    print(CorrAll)
    print(CorrAll_std)
    print(CorrTotal)
    CorrAll_k.append(CorrAll)
    CorrAll_std_k.append(CorrAll_std)
    break
    
print(CorrAll_k)
#print(CorrAll_std_k)
print('mean')
print((np.mean(np.array(CorrAll_k),axis=0)))
print('std')
print((np.mean(np.array(CorrAll_std_k),axis=0)))
print(np.mean(Total))
#yPred=np.array(yPred)
#Y_test=np.array(Y_test)
#print(yPred.shape,Y_test.shape)
#print(ErrAll)
#print(mse(yPred,Y_test))
#for i in range(len(yPred)):
    
    
    

#print(mse(yPred,Y_test))
from scipy.io import savemat
dictMat = {'yPred': yPred,'yTest':Y_test}
os.chdir('/home2/data/navaneetha/')
savemat('EstimatedDavfromAEI008.mat',dictMat)

CC_Values={'CorrAll_k': CorrAll_k, 'CorrAll_std_k': CorrAll_std_k}
os.chdir('/home2/data/navaneetha/speech-emg/EMBC/')
savemat('CC_Values.mat',CC_Values)


#dictMat = {'yTrain1': X_train,'yTrain2':X_train1}
#os.chdir('/home/shankar/Desktop')
#savemat('yTrain.mat',dictMat)
