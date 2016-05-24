%Script to show one example run for Distributed PCA analysis of 
%k-means Clustering algorithm
%Dataset considered is Pendigits of size: N=10992, dim=16 distributed using a
%weighted partitioning method, over a random graph network with n=9,p=0.3.
%We compute clusters(k=10, t=5000) using distributed coreset and Lloyd's kmeans 
%algorithms, of the data reduced to dimension t_vector=[14].

	%Load data
	%P=importdata('pendigits.mat');
    %P=importdata('../data/noisy_circles.csv');
    %P=importdata('../data/noisy_moons.csv');
    P=importdata('../data/spiral.csv');
    Pwclass = P
    K = length(unique(Pwclass(:,3))) % it can be manually set also!
    P = P(:,1:2)
    
	%'Random'graph generation with n=9, p=0.3
    Nnodes = 3;
    G= random_graph_gen(Nnodes, 0.3);    
    fprintf('generated random graph\n');
   
    %partitioning data into 9 local sets using 'weighted' partition method
    [N, dim] = size(P);
    indn=get_partition('weighted', N, Nnodes, sum(G), 1, P);
    
    %Distributed PCA of the data with t_vector = [14]
    %%proj_vector = distributed_pca(P, [14], 9, indn);
    %%lowDim_P = P*proj_vector{1};
    lowDim_P = P
    
    
    %Distributed_coreset construction and lloyd's k-means impementation
    %for the PCA data with k=10, t=10% of the size of the data
    [S,w] = distributed_coreset(lowDim_P, indn, Nnodes, K, floor(0.1*N) );
    [centers_coreset]=lloyd_kmeans(K, S, w);
    
    %% Finding the closest center to each coreset point
    dims = size(S);
    labeledS = zeros(dims(1), dims(2) + 1);
    labeledS(:,1:2) = S;
    
    for i=1:dims(1)
        min_d = inf
        min_c = -1
        for c=1:size(centers_coreset,1)
            
        end
    end
    %%
    
    centers_dim = centers_coreset*proj_vector{1}';
    y = sqDistance(centers_dim, P);
    min_y= min(y);
    kMeansCoresetCost = sum(min_y);
    fprintf('kmeans cost at dim 14=%f\n',kMeansCoresetCost);
    CommunicationCost = size(S,1)*size(S,2)*sum(G(:));
    
    %PCA communication cost
    PCA_comm_cost = (1+ dim)*9 + (14+14*dim)*9;
    CommunicationCost = CommunicationCost + sum(G(:)) * PCA_comm_cost;
    fprintf('coreset_comm_cost at dim 14 =%f\n', CommunicationCost);
    
    
    