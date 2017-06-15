clear all; clc;
%poolsize = 8;
%parpool(poolsize);
%Par parameters are for parallel processing of the files
% ----- update the path to respective directories
label_addr = '/home/siddharthm/scd/vad/train/';
mfcc_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/mfcc_after/train/';
kurt_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/kurt_after/val/';
sfm_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/sfm_after/val/';
mel_feats_addr = '/home/neerajs/work/NEW_REGIME/SID/FEATS/mel_after/train_4Khz/';
linear_feats_addr='/home/neerajs/work/NEW_REGIME/SID/FEATS/linear_after/train/';
EXTRA='/home/siddharthm/scd/feats/gamma/train/';
context_addr = '/home/siddharthm/scd/context/';
% ----- list of files
f=fopen('/home/siddharthm/scd/lists/rawtrainfiles.list');
f=textscan(f,'%s');
len=cellfun('length',f)
type = 'EXTRA';
%context_size = 5;

%parfor i = 1:len
for i = 1:len
[i len]
% read the label file        
temp = load([label_addr f{1}{i}]);
vad=temp.labels % The labels are now stored in the 
% check the feature type
switch(type)
        case 'KURT'
                data_kurt = load([kurt_feats_addr f{1}{i} '.mat']);
                data = data_kurt.normdata';
                op_path = 'kurt';
        case 'SFM'
                data_sfm = load([sfm_feats_addr f{1}{i} '.mat']);
                data = data_sfm.normdata';
                op_path = 'sfm';
        case 'MFCC'
                [data_mfcc,a,b,c,d] = readhtk([mfcc_feats_addr f{1}{i} '.htk']);
                data = data_mfcc;
                op_path = 'mfcc';
        case 'MEL'
                [data_mel,a,b,c,d] = readhtk([mel_feats_addr f{1}{i} '.htk']);
                data = data_mel;
                op_path = 'mel';
        case 'LINEAR'
            [   data_linear,a,b,c,d]=readhtk([linear_feats_addr f{1}{i} '.htk']);
                data=data_linear;
                op_path='linear';
        case 'EXTRA'
                [data_extra,a,b,c,d]=readhtk([EXTRA f{1}{i} '.htk']);
                data=data_extra';
                op_path='context'
end

% The idea is to generate the final datafile, for each file which kind of will include the context
% how the data is created and stored is upto me
% For each file we have many entries in a row, we want to take a certain number of entries <--- Fuck this
% We have input of type, for a file: NFilts X Samples in the File. Fucking take 40 columns. Flatten it in a row major order, and store along with the label.
% Data is now of the form 64 X Number of frames

% Displaying the size of Data input and the the file
size(data)
f{1}{i}
%pause
% read each feature file and make context feature file
nframes = min(size(data,2),length(vad));
%data_write = zeros(nframes,(2*context_size+1)*size(data,1));

temp_matrix=zeros(64,40);
final_matrix=zeros(40*size(data,1),65) %65 columns as 64 filts and 1 label file
start=1;
fin=start+39;
%We want to take 40 such frames, and store this matrix in a row major order
while fin < nframes
        %display([index nframes])
        temp_matrix=data(:,start:fin)
        temp_vector=temp_matrix.reshape()
        start=start+1
end
% save them
writehtk([context_addr op_path '/val/' f{1}{i} '.htk'],data_write,0.11,9); 
%writehtk([context_addr 'labels/val/' f{1}{i} '.htk'],label_write,0.11,9);
end