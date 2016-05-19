function [centers, ind, cost]=single_swap(k, D, w, n_run,max_iter)
%single-swap for k-means, using k-means++ seeding
% input
% k: number of clusters
% D: data matrix, each row is a data point
% optinal input
% w: weights for the points; [] means unweighted
% n_run: number of independent runs; [] means using default value
% max_iter: number of maximum swaps
% output
% centers: cluster centers, each row is a center
% ind: cluster index for points
% cost: kmeans cost for each point

% initialization: 
%   k-means++ seeding
% algorithm: from the paper "A Local Search Approximation Algorithm for k-Means Clustering"
% termination criteria: 
%   the cost decreases by less than 1\% by a single swap
%   or the number of iteration exceeds max_iter
% repeatition:
%   repeat the algo for n_run
%   (default=DEFAULT_N_RUN_FAST for <MAX_POINTS_FOR_FAST points;=DEFAULT_N_RUN_SLOW for >=MAX_POINTS_FOR_FAST points) times, 
%   and select the best run with minimum cost
    
    DEFAULT_N_RUN_FAST=10;
    DEFAULT_N_RUN_SLOW=10;
	MAX_POINTS_FOR_FAST=500000; % if #data points larger than this, the distance matrix may be too large so slow computation is used; 
                                % otherwise, keep the distance matrix in memory for fast computation
    
    if nargin<3 
        w=[];
    end
    if ~isempty(w)
        w(w<0)=0;
    end
    if nargin < 4 || isempty(n_run)
        if size(D, 1)<MAX_POINTS_FOR_FAST
            n_run=DEFAULT_N_RUN_FAST;
        else
            n_run=DEFAULT_N_RUN_SLOW;
        end
    end
    if nargin < 5
        max_iter=500;
    end
    
    [centers, ind, cost]=single_swap_run(k,D,w,max_iter,MAX_POINTS_FOR_FAST);
    for i=2:n_run
        [centers2, ind2, cost2]=single_swap_run(k,D,w,max_iter,MAX_POINTS_FOR_FAST);
        if sum(cost2)<sum(cost)
            centers=centers2;
            ind=ind2;
            cost=cost2;
        end
    end    
    
end


function [centers, ind, cost]=single_swap_run(k,D,w,max_iter,MAX_POINTS_FOR_FAST) 
    if size(D, 1)<MAX_POINTS_FOR_FAST
        [centers, ind, cost]=single_swap_run_fast(k,D,w,max_iter);
    else
        [centers, ind, cost]=single_swap_run_slow(k,D,w,max_iter);
    end    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% slow 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [centers, ind, cost]=single_swap_run_slow(k,D,w,max_iter)    
    [N,~]=size(D);
    min_decrease=0.01;
    if isempty(w)
        w=ones(N,1);
    end        
    
    %k-means++ seeding
    c_ind=kmeanspp(k,D,w);
    c_cost=cost_compute_weighted(D,c_ind,w);
    
    % swap
    iter=1;
    while iter<max_iter
        
        [ii,~,c_ind,c_cost]=find_swap_slow(D,w,c_ind,c_cost,min_decrease);
        if(ii<0)
            break;
        end
        
        iter=iter+1;
    end
    
    centers=D(c_ind,:);
    [d, ind]=min(sqDistance(centers,D));
    ind=ind';
    cost=d'.*w;
    
end

function cost=cost_compute_weighted(D, c_ind,w)
    Dist=sqDistance(D(c_ind,:),D);
    d=min(Dist);
    cost=d*w;
    
end

function [ii,jj,new_ind,new_cost]=find_swap_slow(D,w,c_ind,c_cost,min_decrease)
%find new centers; if not found, ii=jj=-1 and new_ind=c_ind,new_cost=c_cost
    k=length(c_ind);
    N=length(D);
    
    ii=-1;
    jj=-1;
    new_ind=c_ind;
    new_cost=c_cost;
    for i=randperm(k) % permute to avoid bias and repeatition
        for j=randperm(N)
           tmp_ind=c_ind;
           tmp_ind(i)=j;
           tmp_cost=cost_compute_weighted(D,tmp_ind,w);
           if (tmp_cost<(1-min_decrease)*c_cost)
               ii=i;
               jj=j;
               new_ind=tmp_ind;
               new_cost=tmp_cost;
               return;
           end
        end
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fast 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [centers, ind, cost]=single_swap_run_fast(k,D,w,max_iter)
    % equivalent to unweighted
    if ~isempty(w) && all(w==w(1))
        w=[];
    end
    min_decrease=0.01;
    
    Dist=sqDistanceIn(D); % if out of memory for large D, use single_swap_weighted_run_slow
    Dist(Dist<0)=0; % float calculation may lead to some small negative values
        
    %k-means++ seeding
    c_ind=kmeanspp(k,D,w);
    
    tmp_d=min(Dist(c_ind,:));
    if isempty(w)
        c_cost=sum(tmp_d);
    else
        c_cost=tmp_d*w;
    end
    
    % swap
    iter=1;
    while iter<max_iter
        
        [ii,~,c_ind,c_cost]=find_swap_fast(Dist,w,c_ind,c_cost,min_decrease);
        
        if(ii<0)
            break;
        end
        iter=iter+1;
    end
    
    centers=D(c_ind,:);
    [d, ind]=min(Dist(c_ind,:));
    ind=ind';
    if isempty(w)
        cost = d';
    else
        cost= d'.*w;
    end
    
end


function [ii,jj,new_ind,new_cost]=find_swap_fast(Dis,w,c_ind,c_cost,min_decrease)
%find new centers; if not found, ii=jj=-1 and new_ind=c_ind,new_cost=c_cost
    
    k=length(c_ind);
    N=length(Dis);
    
    ii=-1;
    jj=-1;
    new_ind=c_ind;
    new_cost=c_cost;
    
    threshold=(1-min_decrease)*c_cost;
    
    rk=randperm(k);
    rN=randperm(N);
    if isempty(w)
        for i=rk 
            tmp_ind=c_ind;
            tmp_ind(i)=[];
            tmp_d=min(Dis(tmp_ind,:));
            for j=rN
                tmp_cost=sum(min(tmp_d, Dis(j,:)));
                if (tmp_cost<threshold)
                   ii=i;
                   jj=j;
                   new_ind=c_ind;
                   new_ind(i)=j;
                   new_cost=tmp_cost;
                   return;
               end
            end
        end
    else
        for i=rk 
            tmp_ind=c_ind;
            tmp_ind(i)=[];
            tmp_d=min(Dis(tmp_ind,:));
            for j=rN
                tmp_cost=min(tmp_d, Dis(j,:))*w;
                if (tmp_cost<threshold)
                   ii=i;
                   jj=j;
                   new_ind=c_ind;
                   new_ind(i)=j;
                   new_cost=tmp_cost;
                   return;
               end
            end
        end
    end
    
end
