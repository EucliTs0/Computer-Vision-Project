clear all
clc 
close all

run('vlfeat/toolbox/vl_setup')

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

 d_perIm = 300;

  
  %Build vocabulary (bow) using training set
  

opts = {'norm', 'fast', 'floatdescriptors', ...
             'step', 4, ...
             'size', 8, ...
             'geometry', [4 4 8]} ;
%Use the train+validation set to extract descriptors and use a subset of them to create the vocabulary.             
   
  for k = 1:length(fileList_train_val)
       basefilename_train_val = fileList_train_val{k};
       fprintf(1, 'Now reading %s\n', basefilename_train_val);
       imageArray = imread(basefilename_train_val);
       image_gray = rgb2gray(imageArray);
             
       features = extract_voc(image_gray, opts);
       randn('state',0) ;
       rand('state',0) ;
       d_subset = vl_colsubset(1:size(features,2), single(d_perIm));
       features = features(:, d_subset);
       %[~, train_features] = vl_phow(single(image_gray), opts{:});
       visual_words= cat(2, visual_words, features);
  
  
  end
  
  
  %%
  %Build a visual vocabulary using k-means: Set the number of clusters in numClusters variable.
  %save('visual_words_dsift_1000_300sub.mat', 'visual_words');
  numClusters = 500 ;
  
  fprintf('Clustering ...... This may take a while');
  tic;
[centers, assignments] = vl_kmeans(visual_words, numClusters, 'verbose', 'algorithm', 'elkan', 'Initialization', 'PLUSPLUS', 'MaxNumIterations', 50, 'NumRepetitions', 5);
clustering_time = toc;

savefile_1000 = 'centers_500_300sub';
save(savefile_1000, 'centers');

%%
%Encode images using spatial histograms or 'hard' histograms
trainIm_histograms = [];
valIm_histograms = [];
testIm_histograms = [];
forest = vl_kdtreebuild(centers, 'numTrees', 2);
opts_h = {'norm', 'fast', 'floatdescriptors', ...
             'step', 4, ...
             'size', 8, ...
             'geometry', [4 4 8]} ;
 %%
for k = 1:length(fileList_train)
    basefilename_train = fileList_train{k};
    fprintf(1, 'Now reading %s\n', basefilename_train);
    imageArray = imread(basefilename_train);
    image_gray = rgb2gray(imageArray);
    %H = build_hist(im2single(image_gray), forest, centers, opts_h);
    H = build_Spatialhist(im2single(image_gray), forest, centers, opts_h);
    trainIm_histograms = cat(1, trainIm_histograms, H);
    
end
%save('train_histHARD_centers1000.mat', 'trainIm_histograms');

for k = 1:length(fileList_val)
    basefilename_val = fileList_val{k};
    fprintf(1, 'Now reading %s\n', basefilename_val);
    imageArray = imread(basefilename_val);
    image_gray = rgb2gray(imageArray);
    %H = build_hist(im2single(image_gray), forest, centers, opts_h);
    H = build_Spatialhist(im2single(image_gray), forest, centers, opts_h);
    valIm_histograms = cat(1, valIm_histograms, H);
    
end
%save('val_histHARD_centers1000.mat', 'valIm_histograms');

for k = 1:length(fileList_test)
    basefilename_test = fileList_test{k};
    fprintf(1, 'Now reading %s\n', basefilename_test);
    imageArray = imread(basefilename_test);
    image_gray = rgb2gray(imageArray);
    %H = build_hist(im2single(image_gray), forest, centers, opts_h);
    H = build_Spatialhist(im2single(image_gray), forest, centers, opts_h);
    testIm_histograms = cat(1, testIm_histograms, H);
    
end
save('histSPATIAL_centers1000.mat', 'trainIm_histograms', 'valIm_histograms', 'testIm_histograms');





