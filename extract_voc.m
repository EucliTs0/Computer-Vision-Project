function features = extract_voc(I, opts)

[~, features] = vl_dsift(I, opts{:});
features = normalize_features(features);

end

function x = normalize_features(x)
x = bsxfun(@rdivide, x, sqrt(sum(x.^2,1))) ;
end









