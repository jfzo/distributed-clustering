function [S,w]=combined_coreset(k, D, D_w, n , indn, t, constant_approx_algo)
%Build coreset on weighted points
% input
% k: number of clusters
% D: data matrix, each row is a data point
% D_w: weights for data points; [] means unweighted
% n: no of local data sets 
% indn: data partition information
% t: number of sampled points
% optional: 
% constant_approx_algo: the handle of a constant approximation function;
%        should have the form [centers, ind, cost]=constant_approx_algo(k,D,w),
%        where ind are the cluster partition, cost are the costs of the
%        points
% output
% S: the coreset points
% w: the weights for the coreset points    
    if(isempty(D_w))
        D_w=ones(size(D,1),1);
    end

    S_ind = cell(n,1);
    w_ind = cell(n,1);
    
    %constant approximation
    for i=1:n
    	if nargin<7
        		[S_ind{i}, w_ind{i}]=coreset(k,D(indn==i,:), D_w(indn==i,:), floor(t/n)); 
        else
        		[S_ind{i}, w_ind{i}]=coreset(k,D(indn==i,:), D_w(indn==i,:), floor(t/n), constant_approx_algo);
    	end
    end
    S=cell2mat(S_ind);
    w=cell2mat(w_ind);
end
