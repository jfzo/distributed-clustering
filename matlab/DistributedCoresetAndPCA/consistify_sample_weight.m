function cw=consistify_sample_weight(w)
%Ensure that weights must be non-negative values with at least one positive value
%so that it can be used in the function randsample
    if issparse(w)
        w=full(w);
    end
    
    if size(w)==0
        fprintf('empty weights\n');
    end
    
    cw=abs(w);    
    if ~(sum(cw) > 0) || ~all(cw>=0) 
        cw=ones(size(cw));
    end
end