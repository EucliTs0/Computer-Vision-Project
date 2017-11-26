%After you run the creat_BoW.m file and you get your histogram encodings for all the images saved in a .mat file, 
you can load here the .mat file and feed the histograms to a linear SVM, by using homogeneous kernel maps in order to keep the linearity.

load ('histSPATIAL_centers1000.mat');

%Set the svm options
svm_opts = {'Solver', 'sdca', 'Verbose', ...
       'BiasMultiplier', 1, ...
       'Epsilon', 10^-5, ...
       'MaxNumIterations', 100*numel(trainIm_histograms)} ;

psix = vl_homkermap(trainIm_histograms', 1, 'kchi2') ;
psix_test = vl_homkermap(testIm_histograms', 1, 'kchi2') ;
psix_val = vl_homkermap(valIm_histograms', 1, 'kchi2');


%Use the validation set to fix the C hyperparameter.
%We construct 20 linear classifier, one for each class.
count = 1;
best_meanAP = 0;
for c = -2:2
        
    lambda = 1/ ( (2^c)*length(train_labels));
    
    for i = 1:numClasses
        fprintf('Training model for class %s\n', str_train_labels{i}) ;
        y = 2 * ( train_labels(:, i) == 1) - 1 ;
        [w{i}, b{i}, info, S(i,:)] = vl_svmtrain(psix, y', lambda, svm_opts{:});
        scores{i} = w{i}' * psix_val + b{i} ;
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

%Take the best C value and use it to classify the test set.

lambda = 1/ ( best_C*length(train_labels));
    %lambda = 0.003;

    for i = 1:numClasses
        fprintf('Training model for class %s\n', str_train_labels{i}) ;
        y = 2 * ( train_labels(:, i) == 1) - 1 ;
        [w{i}, b{i}, info, S(i,:)] = vl_svmtrain(psix, y', lambda, svm_opts{:});
        scores{i} = w{i}' * psix_test + b{i} ;
        [~,~,info] = vl_pr(test_labels(:, i)', scores{i}) ;
        ap_test(i) = info.ap ;

        fprintf('class %s AP %.2f; \n', str_test_labels{i}, ...
              ap_test(i) * 100) ;
    end
    
    
    meanAP = sum(ap_test) / length(ap_test);

