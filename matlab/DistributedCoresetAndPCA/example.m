%Script to show one example run for Distributed PCA analysis of 
%k-means Clustering algorithm
%Dataset considered is Pendigits of size: N=10992, dim=16 distributed using a
%weighted partitioning method, over a random graph network with n=9,p=0.3.
%We compute clusters(k=10, t=5000) using distributed coreset and Lloyd's kmeans 
%algorithms, of the data reduced to dimension t_vector=[14].

	%Load data
	P=importdata('pendigits.mat');
    
	%'Random'graph generation with n=9, p=0.3
    G= random_graph_gen(9,0.3);    
    fprintf('generated random graph\n');
   
    %partitioning data into 9 local sets using 'weighted' partition method
    [N, dim] = size(P);
    indn=get_partition('weighted', N, 9, sum(G), 1, P);
    
    %Distributed PCA of the data with t_vector = [14]
    proj_vector = distributed_pca(P, [14], 9, indn);
    lowDim_P = P*proj_vector{1};
    
    %Distributed_coreset construction and lloyd's k-means impementation
    %for the PCA data with k=10, t=5000
    [S,w] = distributed_coreset(lowDim_P, indn, 9, 10, 5000);
    [centers_coreset]=lloyd_kmeans(10, S, w);
    
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
    
    
    