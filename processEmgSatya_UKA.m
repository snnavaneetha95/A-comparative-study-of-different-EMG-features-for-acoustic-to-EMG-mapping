function [out1] = processEmgSatya(in)
in = resample(in,5,3);
emg{1,1}=in;
AR = [1 2 3; 4 5 6];
%EMG_data_ln = linenoise_remove_emg(emg,[20 450],AR);
EMG_data_ln =emg;
[b,a]=cheby2(5,40,[3 300]/500);

chan=size(EMG_data_ln{1,1},2);

% % % [h,w]=freqz(emg{1,1}(:,8),1);
% % % temp=EMG_data_ln{1,1};
% % % [h1,w1]=freqz(EMG_data_ln{1,1}(:,8),1);
% % % figure(200);
% % % plot(w/pi*5000,20*log10(abs(h)));hold on;
% % % plot(w1/pi*5000,20*log10(abs(h1)),'r');
% % % 
% % % wavwrite(in(:,8)/abs(max(in(:,8))),10000,'/tmp/in1.wav');
% % % wavwrite(temp(:,8)/abs(max(temp(:,8))),10000,'/tmp/temp1.wav');

out=0*EMG_data_ln{1,1};

[n,wn]=cheb1ord(5/500,15/500,1,40);
[b1,a1]=cheby1(n,1,wn);
for j = 1:chan
    
    signal1 = EMG_data_ln{1,1}(:,j);
    xf = filtfilt(b,a,signal1);
    signal                 = xf.^2;
    
%     % Box filter
%     x                      = filter(ones(350,1)./350,1,signal);
%     EMG_.data_ln{i,2}(:,j) = sqrt(x);
    
    % Gaussian RMS filter
    win=kernels(349);bwin=win.kernel{1};
%     bwin=gausswin(349);
    x                      = filtfilt(b1,a1,signal);
    %temp = x;
    temp = abs(x);
    temp = movmean(temp,10);
%     out(:,j)=temp;
    out(:,j)=temp;
%     out(:,j)=temp;
    %out(:,j)=temp;
    
%     plot([1:length(temp)]/10000,temp);hold on;plot([1:length(temp)]/10000,out(:,j),'r');pause;hold off;
    
end

% % % [h2,w2]=freqz(out(:,8),1);
% % % plot(w2/pi*5000,20*log10(abs(h2)),'k');


out1=out(1:10:end,:);

% % % figure(203);
% % % % % % plot(in(:,8).^2);hold on;plot(temp(:,8).^2,'r');
% % % plot([0:length(out)-1]/10000,out(:,8),'k','linewidth',4);hold on;plot([0:length(out1)-1]/100,out1(:,8),'*-r');
end