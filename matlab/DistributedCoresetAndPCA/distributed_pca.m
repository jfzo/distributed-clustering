function [proj_vector] = distributed_pca(P, t_vector, n, indn)
% PCA on distributed data set
% input:
% P: Original data matrix
% indn: an array of indices; indn(i) is the index of the node that data
%       point i is assigned to
% t_vector: vector of reduced dimension sizes
% n: no. of nodes in the network
% output:
% proj_vector: vector of projection of the data points onto a space with
%              lesser dimensions.
%              Data matrix needs to be multiplied by this matrix to obtain 
%              data in reduced dimension space

    if issparse(P)  
        proj_vector=DistributedPCA_sparse(P, t_vector, n, indn);
    else
        proj_vector=DistributedPCA_dense(P, t_vector, n, indn);
    end
end

function proj_vector=DistributedPCA_dense(P, t_vector, n, indn)
% Distributed PCA on dense data set
% input:
% P: Original data matrix
% indn: an array of indices; indn(i) is the index of the node that data
%       point i is assigned to
% t_vector: vector of reduced dimension sizes
% n: no. of nodes in the network
% output:
% proj_vector: vector of projection of the data points onto a space with
%              lesser dimensions.
%              Data matrix needs to be multiplied by this matrix to obtain 
%              data in reduced dimension space

    [N,d]=size(P);
    
    %Normalize the data
    sumP=sum(P);
    P=P-repmat(sumP/N,[N 1]);
    
    %Local PCA
    D = cell(n,1);
    E = cell(n,1);
    for i=1:n
        % SVD on data from ith node
        [~, D{i}, E{i}] = svd(P(indn==i,:));
    end

    %Global PCA
    t_size = length(t_vector);
    proj_vector = cell(t_size,1);
    for j=1:t_size
        % Generating projection vector for each dimension size
        t=t_vector(j);
        if t>=d
            proj_vector{j}=[];
            continue;
        end

        S = cell(n,1);
        totalS = zeros(d,d);
        for i=1:n
            if isempty(E{i}) || isempty(D{i})
                continue;
            end
            S{i} = E{i}(:,1:t)*D{i}(:,1:t)'*D{i}(:,1:t)*E{i}(:,1:t)';
            totalS = totalS+S{i};
        end
        [vect,~] = eig(totalS);
        proj_vector{j} = vect(:,1:t);
    end
end

function proj_vector=DistributedPCA_sparse(P, t_vector, n, indn)
% Distributed PCA on sparse data set 
% input:
% P: Original data matrix
% indn: an array of indices; indn(i) is the index of the node that data
%       point i is assigned to
% t_vector: vector of reduced dimension sizes
% n: no. of nodes in the network
% output:
% proj_vector: vector of projection of the data points onto a space with
%              lesser dimensions.
%              Data matrix needs to be multiplied by this matrix to obtain 
%              data in reduced dimension space
    
    [N,d]=size(P);
    t_size = length(t_vector);
    
    %Normalize the data
    meanP=full(sum(P))/N;
        
    %Local PCA    
    D = cell(n,1);
    E = cell(n,1);
    for i=1:n
        Pi=full(P(indn==i,:));
        if isempty(Pi)
            D{i}=[]; E{i}=[];
        else
            Pi=bsxfun(@minus,Pi,meanP);
            [~, D{i}, E{i}] = svd(Pi, 'econ');
        end
    end     

    %Global PCA
	proj_vector = cell(t_size,1);
    for j=1:t_size
        t=t_vector(j);
        if t>=d
            proj_vector{j}=[];
            continue;
        end
        
        proj_vector{j}=compute_pc(D, E, n, t, d);        
    end
end

function pc=compute_pc(D, E, n, t, d)
% function to compute the projection vector for sparse matrices
% input:
% D: cell matrix containing diagonal matrix of each node data
% E: cell matrix containing unitary matrix of each node data
% n: no. of nodes in the network
% t: reduced dimension size
% output:
% pc: projection of the data points onto a space with t-dimension.
%     Data matrix needs to be multiplied by this matrix to obtain 
%     data in reduced dimension space.
    
    A=cell(n,1);
    for i=1:n
        t_trunc=min(t,min(size(D{i},2),size(E{i},2)));
        if isempty(D{i}) || isempty(E{i}) || t_trunc<1
            A{i}=[];
        else
            Et=E{i}(:,1:t_trunc);
            Dt=D{i}(1:t_trunc,1:t_trunc);
            A{i}=Et*Dt;
        end
    end
    sigma='lm';
    opts.issym = 1;
    opts.maxit=30;
    [pc,~]=eigs(@DEn,d, t, sigma, opts);
        
    function y = DEn(x)
        y=zeros(size(x));
        for ind=1:n
            if ~isempty(A{ind})
                y = y+ A{ind} * ((A{ind})'*x);
            end
        end
    end
end