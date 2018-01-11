function H = build_Spatialhist(I, forest, centers, opts)

%spatialX and spatialX defines the image region. For example spatialX = [1 1] and spatialY = [1 3]
%means that we compute the histogram on the whole image (spatialX = 1, spatialY = 1) and also on the three vertical
%slices (spatialX = 1, spatialY = 3).

spatialX = [1 1];
spatialY = [1 3];
width = size(I, 2);
height = size(I, 1);
[frames, d] = vl_dsift(I, opts{:});

%L2 normalize the features
d = normalize_features(d);

%%Code for spatial histograms has been obtained from VLFEAT library examples%%

[index , ~] = vl_kdtreequery(forest , centers , d);
%[drop, index] = min(vl_alldist(centers, single(d)), [], 1) ;
for i = 1:length(spatialX)
    %Compute histograms on all regions
    binsx = vl_binsearch(linspace(1,width,spatialX(i)+1), frames(1,:)) ;
    binsy = vl_binsearch(linspace(1,height,spatialY(i)+1), frames(2,:)) ;

  % combine the histograms
    bins = sub2ind([spatialY(i), spatialX(i), length(centers)], ...
                 binsy,binsx,index) ;
    hist = zeros(spatialY(i) * spatialX(i) * length(centers), 1) ;
    hist = vl_binsum(hist, ones(size(bins)), bins) ;
    hists{i} = single(hist ./ sum(hist)) ;
    %hists{i} = hist;
    
    %hists{i} = hists{i} ./ norm(hists{i}); 
end
hist = cat(1,hists{:}) ;
H = hist / sum(hist) ;
H = H';


 
end
 function x = normalize_features(x)
 x = bsxfun(@rdivide, x, sqrt(sum(x.^2,1))) ;
 end




