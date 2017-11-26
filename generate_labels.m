%Generate labels for train images
%Make use of the VOC development kit to generate the labels

numClasses = 20;
imgset_train = 'train';

VOCinit_train;

%Store the classes
str_train_labels = VOCopts.classes;
%Convert string classes to numerical ones
[~, ~, L_train] = unique(VOCopts.classes);
ids_train=textread(sprintf(VOCopts.imgsetpath,imgset_train),'%s');
train_labels = zeros(length(ids_train), numClasses) - 1;

for i = 1:length(ids_train)
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids_train{i}));
    for j = 1:length(rec.objects)
        object_class = rec.objects(j).class;
        K = strfind(VOCopts.classes, object_class);
        object_index = find(~cellfun(@isempty,K));
        train_labels(i, object_index) = 1;
    end
end

%Generate labels for test images
imgset_test = 'test';
VOCinit_test;

%Store the classes
str_test_labels = VOCopts.classes;

%Convert string classes to numerical ones
[~, ~, L_test] = unique(VOCopts.classes);

ids_test=textread(sprintf(VOCopts.imgsetpath,imgset_test),'%s');
test_labels = zeros(length(ids_test), numClasses) - 1;

for i = 1:length(ids_test)
    rec_test=PASreadrecord(sprintf(VOCopts.annopath,ids_test{i}));
    for j = 1:length(rec_test.objects)
        object_class_test = rec_test.objects(j).class;
        K = strfind(VOCopts.classes, object_class_test);
        object_index_test = find(~cellfun(@isempty,K));
        test_labels(i, object_index_test) = 1;
    end
end

%Generate labels for validation set
imgset_val = 'val';

VOCinit_val;

%Store the classes
str_val_labels = VOCopts.classes;
%Convert string classes to numerical ones
[~, ~, L_val] = unique(VOCopts.classes);
ids_val=textread(sprintf(VOCopts.imgsetpath,imgset_val),'%s');
val_labels = zeros(length(ids_val), numClasses) - 1;

for i = 1:length(ids_val)
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids_val{i}));
    for j = 1:length(rec.objects)
        object_class = rec.objects(j).class;
        K = strfind(VOCopts.classes, object_class);
        object_index = find(~cellfun(@isempty,K));
        val_labels(i, object_index) = 1;
    end
end

        
    

