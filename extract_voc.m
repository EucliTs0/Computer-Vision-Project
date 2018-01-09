function features = extract_voc(I, opts)

magnif = 3;
binSize = 8;
I = vl_imsmooth(im2single(I), sqrt((binSize/magnif)^2 - .25));
[~, features] = vl_dsift(I, opts{:});
features = snorm(features);

end

function x = snorm(x)
x = bsxfun(@times, x, 1./max(1e-5,sqrt(sum(x.^2,1)))) ;
end









