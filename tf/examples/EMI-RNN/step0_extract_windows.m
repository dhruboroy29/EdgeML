clc
clear all

% Params (in #samples)
sampRate = 256
class_label = 'Targets'
inputdir = ['/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/Data/Austere/Bora_New_Detector/',class_label];
stride = round(sampRate*2/3)%1.5;
winlen = sampRate*1%2.5;
minlength_secs = 1;

cd(inputdir);
% valid_cuts_dir = ['valid_cuts>=',num2str(minlength_secs),'s'];
% if exist(valid_cuts_dir, 'dir') ~= 7
%     mkdir(valid_cuts_dir);
% end

outdir = ['/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Displacement_Detection/Data/Austere/Bora_New_Detector/winlen_',num2str(winlen),'_stride_',num2str(stride),'/',class_label];
if exist(outdir, 'dir') ~= 7
    mkdir(outdir);
end

list_files = dir('*.data');
tot_length=0;
max_walk_secs = 0;
for l = 1:numel(list_files)
    cur_file = list_files(l).name;
    [~,cur_file_name,~] = fileparts(cur_file);
    cur_file_name;
    [I,Q,N]=Data2IQ(ReadBin([cur_file_name,'.data']));
    Comp=ReadRadar(cur_file);
    
    if length(I)/sampRate < minlength_secs
        continue
    end
    
    cur_walk_secs = length(I)/sampRate;
    if cur_walk_secs > max_walk_secs
        max_walk_secs = cur_walk_secs;
    end
    
    %copyfile(cur_file,valid_cuts_dir);
    L = length(I);
    cut_index=1;
    for k1 = 1:stride:L-winlen+1
        %fprintf('%d to %d    ',k1,k1+winlen);
        % datawin(cut_index,:) = Comp(k1:k1+winlen-1);
        temp = Comp(k1:k1+winlen-1);
        temp_I = I(k1:k1+winlen-1);
        temp_Q = Q(k1:k1+winlen-1);
        Data_cut = zeros(1,2*(k1+winlen-1-k1+1));
        Data_cut(1:2:length(Data_cut)-1) = temp_I;
        Data_cut(2:2:length(Data_cut)) = temp_Q;
        %3 lines added by neel to output each sliding window as cut
        %sliding_win = zeros(1,length(temp));
        %sliding_win = temp;
        WriteBin([outdir,'/',cur_file_name,'_',class_label,'_',num2str(k1),'_to_',num2str(k1+winlen),'.data'],Data_cut);
        
        
        % Flatten as Re1 Img1 Re2 Img2....
        tempmat = [real(temp); -1*imag(temp)]; % -1 to correct Hermitian conjugate
        datawin(cut_index,:) = tempmat(:)';
        cut_index = cut_index + 1;
    end
    
    if exist('datawin','var') == 1
        outfile = fullfile(outdir,[cur_file_name,'.csv']);
        %csvwrite(outfile, datawin);
    
%     outsubdir = fullfile(outdir, cur_file_name);
%     if exist(outsubdir, 'dir') ~= 7
%         mkdir(outsubdir);
%     end
%     outfile_real = fullfile(outsubdir,[cur_file_name,'_real.csv']);
%     outfile_imag = fullfile(outsubdir,[cur_file_name,'_imag.csv']);
%     csvwrite(outfile_real, real(datawin));
%     csvwrite(outfile_imag, imag(datawin));
    
        tot_length = tot_length + size(datawin,1);
        clear datawin
    end
end