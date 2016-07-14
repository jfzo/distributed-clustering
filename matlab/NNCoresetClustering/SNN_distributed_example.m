function [knn, snn_graph] = SNN_distributed_example(Z, k)
    % sparsified similarity matrix
    knn = snn_dd(Z, k);
    % snn graph
    N = size(Z,1);
    snn_graph = cell(N-1,1);
    for i = 1:N-1
        knn_i = knn{i};
        snn_i = zeros(N-i, 'int8');
        parfor j = (i+1):N
            snn_i(j) = length(intersect(knn_i, knn{j}));
        end
        snn_graph{i} = snn_i;
    end
end

function [topk] = get_topk_positions(v, k)
    [ ~ , neighs ] = sort(v, 'ascend');
    topk = neighs(2:k+1); % the first item is always the point itself
end

function [knn] = snn_dd_par(P, k)
    % P is the data matrix
    N = size(P,1);
    A = pdist2(P,P);

    knn = cell(N, 1);
    parfor i = 1:N
        knn{i} = get_topk_positions(A(i,:), k);
    end
end

function [knn] = snn_dd(P, k)
    % returns a k+1 x N matrix with the indices 
    % of the smallest distances in each column
    % I(:, i ) denotes the k+1 elements with the smallest 
    % distances (including i )
    N = size(P,1);
    [~, I] = pdist2(P, P, 'euclidean', 'Smallest', k+1);
    knn = cell(N, 1);
    parfor i = 1:N
        nn_i = I(:, i);
        knn{i} = nn_i(2:end);
    end
end
