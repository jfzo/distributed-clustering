function [S,w]=coreset(k, D, D_w, t, constant_approx_algo)
%Build coreset on weighted points
% input
% k: number of clusters
% D: data matrix, each row is a data point
% D_w: weights for data points; [] means unweighted
% t: number of sampled points
% optional: 
% constant_approx_algo: the handle of a constant approximation function;
%        should have the form [centers, ind, cost]=constant_approx_algo(k,D,D_w),
%        where ind are the cluster partition, cost are the costs of the
%        points
% output
% S: the coreset points
% w: the weights for the coreset points
    if size(D,1)<=0
		S=[]; w=[]; return;
    end
    % constant approximation 
    if nargin<5
        [centers, ind, cost]=single_swap(k,D,D_w);
    else
        [centers, ind, cost]=constant_approx_algo(k,D,D_w); 
    end
    
	if isempty(D_w)
	    [S,w]=coreset_by_solution(D, [], t,centers, ind, cost);
	else
		[S,w]=coreset_by_solution(D, D_w, t,centers, ind, cost.*D_w);
	end
    
    clear D;
end

function [S,w]=coreset_by_solution(D, D_w, t,centers, ind, cost)
%Build coreset with pre-computed constant approximation solution
% input
% D: data matrix, each row is a data point
% D_w: the weights of the input points; [] means unweighted
% t: number of sampled points
% centers: an array of centers in the approximation solution 
% ind: an array of partition index in the approximation solution
% cost: an array of costs in the approximation solution; for weighted
%           points, this array should be the weighted costs
% optional input
% output
% S: the coreset points
% w: the weights for the coreset points
    
    [N,d]=size(D);
    
    if isempty(D_w)
        D_w=ones(N,1);
    end
    
    % sample
	sample_weight=consistify_sample_weight(cost);
    S_ind=randsample(N,t,true,sample_weight);
    S=zeros(t+size(centers, 1), d);
    S(1:t,:)=D(S_ind,:);
    S((t+1):t+size(centers, 1),:)=centers;
    
    % weights
    w=zeros(t+size(centers, 1), 1);
	avg_cost=sum(cost)/length(cost);
	w(1:t)=(sum(cost)/t) ./ (cost(S_ind)+0.00001*avg_cost);% +0.00001*avg_costi to avoid 0
	
    wS=w(1:t);
    for i=1:size(centers, 1)
        imap=ismember(S_ind, find(ind==i));
        w(t+i)=sum(D_w(ind==i))- sum(wS(imap));
    end
end