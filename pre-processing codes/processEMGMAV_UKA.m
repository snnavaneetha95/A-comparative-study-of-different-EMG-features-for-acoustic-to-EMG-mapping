function [out1] = processEMGMAV(in)
in = resample(in,5,3);
emg{1,1}=in;
AR = [1 2 3; 4 5 6];
%EMG_data_ln = linenoise_remove_emg(emg,[20 450],AR);
EMG_data_ln = emg;
chan=size(EMG_data_ln{1,1},2);
[b,a]=cheby2(5,40,[3 300]/500);

%Wp = [10 2000]/5000;                                    % Passband Frequencies (Normalised)
%Ws = [9 2005]/5000;                                    % Stopband Frequencies (Normalised)
%Rp = 10;                                                % Passband Ripple (dB)
%Rs = 50; 
%[n,wn]=cheb1ord(Wp,Ws,1,40);
%[b,a]=cheby1(n,1,wn);

for j = 1:chan
    signal1 = EMG_data_ln{1,1}(:,j);
    xf = filtfilt(b,a,signal1);
    signal                 = abs(xf);
    signal = movmean(signal,10);
    out(:,j)=signal;
end

out1=out;
out1=out(1:10:end,:);
end
