function [centers, ind, cost]=lloyd_kmeans(k,D,w,n_run)
%Lloyd's method for k-means, using k-means++ seeding
% input
% k: number of clusters
% D: data matrix, each row is a data point
% optinal input
% w: weights for the data points; [] means unweighted
% n_run: number of runs
% output
% centers: cluster centers
% ind: cluster index for points
% cost: kmeans cost for each point

% initialization: 
%   k-means++ seeding
% termination criteria: 
%   over three consecutive stages, the average distortion decreases by less than 1%
%   or the number of iterations exceeds 500;
%   but the minimum number of iterations is 100
% repeatition:
%   repeat the algo for n_run(default=10) times, and select the best run with minimum cost

    if nargin<3
        w=[];
    end
    if nargin<4
        n_run=10;
    end
    
    [centers, ind, cost]=lloyd_kmeans_run(D,k,w);
    for i=2:n_run
        [centers2, ind2, cost2]=lloyd_kmeans_run(D,k,w);
        if sum(cost2)<sum(cost)
            centers=centers2;
            ind=ind2;
            cost=cost2;
        end
    end
    
end

% w=[] means unweighted
function [centers, ind, cost]=lloyd_kmeans_run(D,k,w)
    min_decrease=0.01;
    min_iter=100;
    max_iter=500;
    
    %k-means++ seeding
    init_c_ind=kmeanspp(k,D,w);    
	
    % lloyd step
    cost_iter=zeros(max_iter,1);
    centers_iter=cell(max_iter,1);
    iter=1;
    c=D(init_c_ind,:);
    while iter<max_iter
        centers_iter{iter}=c;
        
        % partition
        Dist=sqDistance(c,D);
        [dist, part_ind]=min(Dist);
        
        
        if isempty(w)
            cost_iter(iter)=sum(dist);
        else
            cost_iter(iter)=  dist * w;
        end
        
        % move to centroid
        for i=1:k
            if isempty(w)
                c(i,:)=mean(D(part_ind==i,:));
                if sum(part_ind==i)==0
                    c(i,:)=D(init_c_ind(i),:);
                end
            else
                c(i,:)=weighted_mean(D(part_ind==i,:),w(part_ind==i,:));
                if sum(part_ind==i)==0 || sum(w(part_ind==i,:))==0
                    c(i,:)=D(init_c_ind(i),:);
                end
            end
        end
        
        % check termination
        if (iter>min_iter && (cost_iter(iter)-cost_iter(iter-3))/cost_iter(iter-3) < min_decrease)
            break;
        end
        
        iter=iter+1;
    end
    
    [~,ind_c]=min(cost_iter(1:(iter-1)));
    centers=centers_iter{ind_c};
    
    Dist=sqDistance(centers,D);
    [cost, ind]=min(Dist);
    cost=cost';
    if ~isempty(w)
        cost=cost.*w;
    end
    
end

% w: weights, a column vector 
% data: data matrix, each row is a point
function wmean=weighted_mean(data,w)
    wmean=w'*data/sum(w);
    if sum(w)==0
        wmean=zeros(1,size(data,2));
    end
end