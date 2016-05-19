function [Sm,wm,com_m]=merge_coreset_on_tree(T, D, indn, k, n, t, constant_approx_algo)
%Build coreset using the approach in the path-aware algorithm in the paper
%"Approximate Clustering on Distributed Data Streams" 

% input
% T: tree
% D: data
% indn: partition of data on to nodes
% k: number of clusters
% n: number of nodes in the distributed setting
% t: number of sampled points
% optional: 
% constant_approx_algo: the handle of a constant approximation function;
%        should have the form [centers, ind, cost]=constant_approx_algo(D,w,k),
%        where ind are the cluster partition, cost are the costs of the
%        points
% output
% Sm, wm: coreset
% com_m: communcation cost for computing the coreset
    
    com_m=0;
    
    coreset_status =zeros(n,1); % -1 for waiting for children coresets; 0 for ready; 1 for constructed
    for i=1:n
        if sum(T(i,:))~=0
            coreset_status(i)=-1;
        end
    end
    coresets_points=cell(n,1);
    coresets_weights=cell(n,1);
    
    last_ind=0;
    while sum(coreset_status==1)~=n
        for i=1:n
            if coreset_status(i)==0
                children=find(T(i,:)~=0);
                
                % collect data
                union_set=D(indn==i,:);
                union_set_weight=ones(size(union_set,1),1);
                for j=1:length(children)
                    child=children(j);
                    union_set=vertcat(union_set, coresets_points{child});
                    union_set_weight=vertcat(union_set_weight,coresets_weights{child});
                    
                    com_m=com_m+length(coresets_weights{child});
                end
                
                % build coreset
                if nargin<7
                    [S,w]=coreset(k,union_set, union_set_weight, t);
                else
                    [S,w]=coreset(k,union_set, union_set_weight, t, constant_approx_algo);
                end
                
                coresets_points{i}=S;
                coresets_weights{i}=w;
                
                coreset_status(i)=1;
                
                last_ind=i;
            end
        end
        % update the status
        for i=1:n
            if coreset_status(i)==-1
                children=find(T(i,:)~=0);
                if sum(coreset_status(children))==length(children)
                    coreset_status(i)=0;
                end
            end
        end
    end
    Sm=coresets_points{last_ind};
    wm=coresets_weights{last_ind};
    
end
