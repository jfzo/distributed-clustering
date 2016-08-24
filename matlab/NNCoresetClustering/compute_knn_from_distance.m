function [knn, snn_graph] = compute_knn_from_distance( Z, k )
%COMPUTE_KNN_FROM_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

% sparsified similarity matrix
    display('Starting k-near neighbor computation.');
    %knn = snn_dd(Z, k);
    
    N = size(P,1);
    [~, I] = pdist2(P, P, 'euclidean', 'Smallest', k+1);
    knn = cell(N, 1);
    parfor i = 1:N
        nn_i = I(:, i);
        knn{i} = nn_i(2:end);%filters point i
    end
    
    display('k-near neighbor computed.');
    
    % snn graph
    display('Starting SNN graph computation.');
    N = size(Z,1);
    snn_graph = cell(N-1,1);
    
    for i = 1:N-1
        knn_i = knn{i};
        snn_i = zeros(N-i,1, 'int8');
        parfor j = (i+1):N
            %snn_i(j-i) = length(intersect(knn_i, knn{j}));
            snn_i(j-i) = length(my_intersect(knn_i, knn{j}));
        end
        snn_graph{i} = snn_i;
    end
   display('SNN graph built.');
end
