function [out1] = processEmg_navaneetha_UKA(in)
[b,a]=cheby2(5,40,[3 300]/500);
[b1,a1]=cheby2(5,40,20/500);
in = resample(in,5,3);
emg{1,1}=in;
AR = [1 2 3; 4 5 6];
%EMG_data_ln = linenoise_remove_emg(emg,[20 450],AR);
EMG_data_ln = emg;
chan=size(EMG_data_ln{1,1},2);
out=0*EMG_data_ln{1,1};

for j = 1:chan
    signal = EMG_data_ln{1,1}(:,j);
    xf = filter(b,a,signal);
    xf_env=abs(hilbert(xf));
    xf_env_f=filtfilt(b1,a1,xf_env);
    out(:,j) = xf_env_f;
   
    
end
%close all
%plot(xf,'r');hold on;plot(xf_env,'g');plot(xf_env_f,'b');
%legend('bandpassed','hilbert','5hz-bandpassed')
out1=out(1:10:end,:);

end
