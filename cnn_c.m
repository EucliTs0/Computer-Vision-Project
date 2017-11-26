clear all
clc 
close all

%mex -setup C++
run('vlfeat/toolbox/vl_setup')
%mex -setup C++
run('matconvnet/matlab/vl_compilenn')
%run('matconvnet/matlab/vl_setupnn')

train_Folder = 'VOC2007_train/JPEGImages'; %In the train folder, the validation set
%is included, so we have to separate it.

test_Folder = 'VOC2007_test/JPEGImages';

%Make use of the VOCdevkit in order to extract the labels
imgset_trainval = 'trainval';
VOCinit_train;
%Load the names of images (train + val)
ids_trainval=textread(sprintf(VOCopts.imgsetpath,imgset_trainval),'%s');

%Generate labels for all sets (train, val, test);
generate_labels;

visual_words = [];
visual_words_train = [];
vw_hist = [];
visual_words_test = [];

dirData_train = dir(train_Folder);      %# Get the data for the current directory
dirData_test = dir(test_Folder);
dirIndex_train = [dirData_train.isdir];  %# Find the index for directories
dirIndex_test = [dirData_test.isdir];
fileList_train = {dirData_train(~dirIndex_train).name}';  %'# Get a list of the files
fileList_test = {dirData_test(~dirIndex_test).name}';
fileList_train_val = fileList_train;

fileList_train_idx = regexp(fileList_train, '\d*', 'Match');
train_imNames = cell(1, length(fileList_train_idx));
for i = 1:length(fileList_train_idx)
    train_imNames{i} = cell2mat(fileList_train_idx{i});
end

train_imNames = train_imNames';
train_imNames = str2num(cell2mat(train_imNames));
val_imNames = str2num(cell2mat(ids_val));
[rowsF, colsF, valsF] = find(val_imNames == train_imNames');

%Separate the validation set
fileList_val = fileList_train(colsF);
fileList_train(colsF) = [];


                                          
                                          
%%
model_dir = 'deepeval-encoder/models/CNN_S';

fprintf('%s\n', model_dir);

param_file = sprintf('%s/param.prototxt', model_dir);
model_file = sprintf('%s/model', model_dir);

average_image = 'deepeval-encoder/models/mean.mat';

use_gpu = false;

if use_gpu
    featpipem.directencode.ConvNetEncoder.set_backend('cuda');
    % can optionally set gpu device id using the second parameter:
    % featpipem.directencode.ConvNetEncoder.set_backend('cuda', 0);
end
encoder.augmentation = 'aspect_corners';
encoder.augmentation_collate = 'none';

% initialize an instance of the ConvNet feature encoder class
encoder = featpipem.directencode.ConvNetEncoder(param_file, model_file, ...
                                                average_image, ...
                                                'output_blob_name', 'fc7');

%%
train_features = [];

for k = 1:length(fileList_train)
       basefilename_train = fileList_train{k};
       fprintf(1, 'Now reading %s\n', basefilename_train);
       imageArray = imread(basefilename_train);
       imageArray = featpipem.utility.standardizeImage(imageArray);
       
       features = encoder.encode(imageArray);
       features = features';
       
       train_features = cat(1, train_features, features);
  
  
end


val_features = [];

for k = 1:length(fileList_val)
       basefilename_val = fileList_val{k};
       fprintf(1, 'Now reading %s\n', basefilename_val);
       imageArray = imread(basefilename_val);
       imageArray = featpipem.utility.standardizeImage(imageArray);
       
       features = encoder.encode(imageArray);
       features = features';
       
       val_features = cat(1, val_features, features);
       
      
end

test_features = [];
for k = 1:length(fileList_test)
       basefilename_test = fileList_test{k};
       fprintf(1, 'Now reading %s\n', basefilename_test);
       imageArray = imread(basefilename_test);
       imageArray = featpipem.utility.standardizeImage(imageArray);
       
       features = encoder.encode(imageArray);
       features = features';
       
       test_features = cat(1, test_features, features);
  
  
end

save('features_CNN_S_AUG.mat', 'train_features', 'val_features', 'test_features');

%%
svm_opts = {'Solver', 'sdca', 'Verbose', ...
       'BiasMultiplier', 1, ...
       'Epsilon', 10^-5, ...
       'MaxNumIterations', 100*numel(train_features)} ;


count = 1;
best_meanAP = 0;
for c = -2:2
        
    lambda = 1/ ( (2^c)*length(train_labels));
    

    for i = 1:numClasses
        fprintf('Training model for class %s\n', str_train_labels{i}) ;
        y = 2 * ( train_labels(:, i) == 1) - 1 ;
        [w{i}, b{i}, info, S(i,:)] = vl_svmtrain(train_features', y', lambda, svm_opts{:});
        scores{i} = w{i}' * val_features' + b{i} ;
        [~,~,info] = vl_pr(val_labels(:, i)', scores{i}) ;
        ap(i) = info.ap ;

        fprintf('class %s AP %.2f; \n', str_val_labels{i}, ...
              ap(i) * 100) ;
    end
    avg_pr{i} = ap; 
    
    mean_AP(count) = sum(ap)*100/numClasses;
    if mean_AP(count) > best_meanAP
        best_meanAP = mean_AP(count);
        best_C = 2^c;
    end
    count = count + 1;
end
%%

lambda = 1/ ( best_C*length(train_labels));
    
    for i = 1:numClasses
        fprintf('Training model for class %s\n', str_train_labels{i}) ;
        y = 2 * ( train_labels(:, i) == 1) - 1 ;
        [w{i}, b{i}, info, S(i,:)] = vl_svmtrain(train_features', y', lambda, svm_opts{:});
        scores{i} = w{i}' * test_features' + b{i} ;
        [~,~,info] = vl_pr(test_labels(:, i)', scores{i}) ;
        ap_test(i) = info.ap ;

        fprintf('class %s AP %.2f; \n', str_test_labels{i}, ...
              ap_test(i) * 100) ;
    end
    
    
    meanAP = sum(ap_test) / length(ap_test);
