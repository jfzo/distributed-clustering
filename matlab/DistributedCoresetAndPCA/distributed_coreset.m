function [S,w,portion_ind]=distributed_coreset(D,indn,n,k,t,constant_approx_algo)
%Build distributed coreset
% input
% k: number of clusters
% D: data matrix, each row is a data point
% indn: partition data points into n parts
% n: number of partitions (number of nodes in the distributed setting)
% t: number of sampled points
% optional: 
% constant_approx_algo: the handle of a constant approximation function;
%        should have the form [centers, ind, cost]=constant_approx_algo(k,D),
%        where ind are the cluster partition, cost are the costs of the
%        points
% output
% S: the coreset points
% w: the weights for the coreset points
% portion_ind: indicator the index of the node that contains the coreset point

    % constant approximation solutions for local data
    centers=cell(n,1);
    ind=cell(n,1);
    cost=cell(n,1);
    total_cost=0;
    for i=1:n
        if nargin<6
            [centers{i}, ind{i}, cost{i}]=single_swap(k,D(indn==i,:));
        else
            [centers{i}, ind{i}, cost{i}]=constant_approx_algo(k,D(indn==i,:));
        end
        total_cost=total_cost+sum(cost{i});        
    end
    
    [S,w,portion_ind]=distributed_coreset_by_solution(D,indn,n, t, centers, ind, cost, total_cost);
    
end


function [S,w,portion_ind]=distributed_coreset_by_solution(D,indn,n, t, centers, ind, cost, total_cost)
%Build distributed coreset with pre-computed constant approximation
%solution
% input
% centers: a cell, each cell contains an array of centers in the
%   approximation solution for the corresponding local data
% ind: a cell, each cell contains an array of partition index in the
%   approximation solution for the corresponding local data
% cost: a cell, each cell contains an array of costs in the
%   approximation solution for the corresponding local data
% total_cost: the sum of all the local costs
% D: data matrix, each row is a data point
% indn: partition data points into n parts
% n: number of partitions (number of nodes in the distributed setting)
% k: number of clusters
% t: number of sampled points
% output
% S: the coreset points
% w: the weights for the coreset points
% portion_ind: indicator the index of the node that contains the coreset point

    S=[]; w=[]; portion_ind=[];
    
    for i=1:n
      [Si,wi]=distributed_coreset_on_node(D(indn==i,:),ind{i},ceil(sum(cost{i})/total_cost*t),t,centers{i},cost{i},total_cost);
      S=vertcat(S,Si);
      w=vertcat(w,wi);
      portion_ind=vertcat(portion_ind, i*ones(length(wi),1));
    end
    
    clear D;
end

function [Si,wi]=distributed_coreset_on_node(Di,indi,ti,t, centersi,costi,total_cost)
%Build distributed coreset on a given node
% input
% centersi: contains an array of centers in the approximation solution 
%           for the local data in the node
% indi: contains an array of partition index in the approximation solution
%       for the the local data
% costi: contains an array of costs in the approximation solution
%        for the corresponding local data
% total_cost: the sum of all the local costs
% Di: data points in the node
% t: number of sampled points
% output
% Si: the coreset points
% wi: the weights for the coreset points

    % sample
    [Ni,d]=size(Di);
	if Ni<=0
		Si=[]; wi=[]; return;
    elseif Ni<=ti
        Si=Di;
        wi=ones(Ni,1);
        return;
	end
	
	sample_weight=consistify_sample_weight(costi);
    Si_ind=randsample(Ni,ti,true,sample_weight);
    Si=zeros(ti+size(centersi, 1), d);
    Si(1:ti,:)=Di(Si_ind,:);
    Si((ti+1):ti+size(centersi, 1),:)=centersi;
    
    % weights
    wi=zeros(ti+size(centersi, 1), 1);
    avg_costi=sum(costi)/length(costi);
    wi(1:ti)=(total_cost/t) ./ (costi(Si_ind)+0.00001*avg_costi); % +0.00001*avg_costi to avoid 0
    wSi=wi(1:ti);
    for i=1:size(centersi, 1)
        imap=ismember(Si_ind, find(indi==i));
        wi(ti+i)=sum(indi==i)- sum(wSi(imap));
    end
    
end