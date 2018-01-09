function H = build_hist(I, forest, centers, opts)

%Dense SIFT features
[~, d] = vl_dsift(I, opts{:});

%L2 normalize the SIFt features
d = normalize_features(d);
   
[index , ~] = vl_kdtreequery(forest , centers , d);

 %Compute the image histogram 
 H = vl_binsum(zeros(length(centers),1), 1, double(index)) ;
 
 %L1 normalize the histogram
 H = H ./ sum(H);
 
 H = H';


end
 function x = normalize_features(x)
 x = bsxfun(@rdivide, x, sqrt(sum(x.^2,1))) ;
 end


