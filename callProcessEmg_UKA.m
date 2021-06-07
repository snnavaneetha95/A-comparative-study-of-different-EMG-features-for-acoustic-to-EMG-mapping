clc;
clear all;
close all;
SUBJ='C:\Users\Navaneetha\Documents\speech-emg\phone\AudibleUKA\';
NChannel=6;
method = 'MAV';
mkdir([SUBJ method]);
%mkdir([SUBJ '/wav']);
allfiles=dir(['C:\Users\Navaneetha\Documents\speech-emg\AudibleUKA\splicedArray_e07*.mat']);
for i=1:length(allfiles)
    file=allfiles(i).name;
    data=load([allfiles(i).folder '\' file]);
    %wav=data(:,end);
    data = data.ADC_modified;
    %data=data.data;
    data = double(data);
    data = data';
    data=data(:,1:NChannel);
    data=processEMGMAV_UKA(data);
    for k=1:size(data,2)
	for j=1:size(data,1)   
            data(j,k) = data(j,k)./max(data(:,k));
        end
    end
    %plot(data(:,1))
    save([SUBJ method '\' file],'data');
    %wav=wav/max(abs(wav))*.95;
    %audiowrite([SUBJ '/wav/' file(1:end-3) 'wav'],wav,10000);
end
