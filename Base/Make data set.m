clc
close all
clear all
warning off
file_dir='G:\';
csv_dir='G:\未改变方位8000\ov1_split1\desc_ov1_split1\';
out_dir='G:\未改变方位8000\ov1_split1\wav_ov1_split1\';
csv_files=dir(csv_dir);
csv_files=csv_files(3:end);
len_files=length(csv_files);
for i=1:len_files
    files_name=csv_files(i).name; % test_0_desc_30_300.csv
    files_dir=[csv_dir,files_name]; % G:\自制数据集\desc_ov1_split1\test_0_desc_30_300.csv
    [a,txt,raw]=xlsread(files_dir);
    start_time=a(:,1);
    end_time=a(:,2);
    audio=zeros(320000,1); %1323000  8000*30=240000  44100*30
    s=[];
    Fs=8000;
    for j=1:length(start_time)
        wav_name=strcat(file_dir,raw{j+1,1});
        [wave fs]=audioread(wav_name);
        if fs~=8000
            [P,Q]=rat(8000/fs);
            wave=resample(wave,P,Q);  % 重采样是要改变语音长度的
        end
        sample_start=start_time(j)*Fs;
        sample_end=end_time(j)*Fs;
        len=length(wave);
        s=audio(ceil(sample_start):ceil(sample_start)+len-1);
        audio(ceil(sample_start):ceil(sample_start)+len-1)=s+wave;
%         s=audio(ceil(sample_start):ceil(sample_end)-1);
%         audio(ceil(sample_start):ceil(sample_start)+len-1)=s+wave;
        clear sample_start sample_end s
    end
    audio=audio(1:240000);
    output_dir=[out_dir,csv_files(i).name(1:(end-4)),'.wav'];
    audiowrite(output_dir,audio,Fs)
end
