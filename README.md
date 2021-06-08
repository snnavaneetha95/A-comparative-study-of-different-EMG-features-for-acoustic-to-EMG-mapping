We explored different EMG feature extraction methods, which are used in context of speech tasks and non-speech tasks.

Non-Speech Temporal(NST) features: MAV, RMS, DAV and LFB

EMG features used in speech context: LFM, LFP, HFP, HFZCR and HFRM(these are called time domain(TD) features).

Along with above features we propose a novel Hilbert features.

For the Speech to EMG mapping task, we choose MFCC features for speech representations and all the above mentioned features are experimented for EMG.
Experiments are done with individual features and with combination.

We also reconstruct original EMG signal from its features using a CNN-BLSTM network.
Following experiments with features are done
a.	LFM+LFP+HFP+HFZCR+HFRM
b.	LFM+LFP+HFP+HFRM
c.	LFM+LFP+HFP+HFZCR+HFRM+Hilbert
d.	LFM+LFP+HFP+HFRM+Hilbert 
 
Description about the codes:
pre-processing codes/callProcessEmg_UKA.m  this code calls the matlab functions of feature extraction methods
pre-processing codes/processEMGMAV_UKA.m  MAV method
pre-processing codes/processEMGRMS_UKA.m  RMS method
pre-processing codes/processEMGDAV_UKA.m  DAV method
pre-processing codes/processEMGLFB_UKA.m  LFB method
pre-processing codes/processEMGHilbert_UKA.m  Hilbert method
there is a function get_emg_features, which generates 5 Time Domain features
BLSTMModelMono.py   To train the model with the given input MFCC and output EMG(NST  or Hilbert) feature
BLSTMModel_TD.py  To train the model with the given input MFCC and output EMG(5 TD) features, there is a function get_emg_features, which generates 5 Time Domain features.

TDplusHilbertToRawEMG2inputs.py  To train the model, which is used to reconstruct the original signal from TD and Hilbert features.
TDtoRawEMG.py  To train the model, which is used to reconstruct the original signal from TD features.




