function centers=kmeanspp(k, D, w)
%k-means++ seeding
% input
% D: data matrix, each row is a data point
% k: number of clusters
% optinal input
% w: weights for the points; [] means unweighted
% output
% centers: cluster centers, each row is a center
  
    if nargin<3
        w=[];
    end
    
	[N,~]=size(D);
    centers=zeros(k,1);
    centers(1)=ceil(rand(1,1)*N);
    for i=2:k
        y=sqDistance(D(centers(1:(i-1)),:), D);
        if i==2
            dd=y;
        else
            dd=min(y);
        end
        if isempty(w)
            sample_weight=consistify_sample_weight(dd);
            centers(i)=randsample(N,1,true,sample_weight);
        else
			sw=dd.*w';
            sample_weight=consistify_sample_weight(sw);
            centers(i)=randsample(N,1,true,sample_weight);
        end
    end
end