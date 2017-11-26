function H = build_hist(I, forest, centers, opts)

[~, d] = vl_dsift(I, opts{:});
d = normalize_features(d);
   
[index , ~] = vl_kdtreequery(forest , centers , d);

 H = vl_binsum(zeros(length(centers),1), 1, double(index)) ;
 
 H = H ./ sum(H);
 
 H = H';


end
 function x = normalize_features(x)
 x = bsxfun(@rdivide, x, sqrt(sum(x.^2,1))) ;
 end


