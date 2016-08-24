function [ snn_graph ] = compute_knn_snn_from_similarity( S, k )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    % sparsified similarity matrix
    display('Starting k-near neighbor computation.');
    knn = snn_dd(S, k); % cell array of length N with a length k vector in each cell position
    display('k-near neighbor computed.');
    
    % snn graph
    display('Starting SNN graph computation.');
    N = size(S,1);
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

function [knn] = snn_dd(P, k)
    % P is a similarity matrix
    % returns a k+1 x N matrix with the indices 
    % of the smallest distances in each column
    % I(:, i ) denotes the k+1 elements with the smallest 
    % distances (including i )
    N = size(P,1);
    
    %[~, I] = pdist2(P, P, 'euclidean', 'Smallest', k+1);
    knn = cell(N, 1);
    parfor i = 1:N
        [~,ix] = sort(P(i,:), 'descend');
        knn{i} = ix(2:k+1);%filters point i
    end
end