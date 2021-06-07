function [out1] = processEMGDavella(in)

in = resample(in,5,3);
emg{1,1}=in;
AR = [1 2 3; 4 5 6];
%EMG_data_ln = linenoise_remove_emg(emg,[20 450],AR);
EMG_data_ln = emg;
chan=size(EMG_data_ln{1,1},2);
%out=0*EMG_data_ln{1,1};
[b,a]=cheby2(5,40,[3 300]/500);

%[n,wn]=cheb1ord(5/500,15/500,1,40);
%[b,a]=cheby1(n,1,wn);
 %bp = fir1(50,[10/300 100/300],'bandpass');
 hp = fir1(50,20/500,'low');
for j = 1:chan
    signal1 = EMG_data_ln{1,1}(:,j);
    xf = filtfilt(b,a,signal1);
    signal                 = sqrt(xf.^2);
    
%     % Box filter
%     x                      = filter(ones(350,1)./350,1,signal);
%     EMG_.data_ln{i,2}(:,j) = sqrt(x);
    
    % Gaussian RMS filter
   % win=kernels(349);bwin=win.kernel{1};
%     bwin=gausswin(349);
    %x                      = filtfilt(b,a,signal);
    %x = filter(bp,1,signal);
    x = filter(hp,1,signal);
    temp = abs(x);
    temp = movmean(temp,10);
%   out(:,j)=temp;
    out(:,j)=temp;
    
%     plot([1:length(temp)]/10000,temp);hold on;plot([1:length(temp)]/10000,out(:,j),'r');pause;hold off;
    
%end

% % % [h2,w2]=freqz(out(:,8),1);
% % % plot(w2/pi*5000,20*log10(abs(h2)),'k');

%end
end
out1= out;
out1=out(1:10:end,:);
end


% % % figure(203);
% % % % % % plot(in(:,8).^2);hold on;plot(temp(:,8).^2,'r');
% % % plot([0:length(out)-1]/10000,out(:,8),'k','linewidth',4);hold on;plot([0:length(out1)-1]/100,out1(:,8),'*-r');
