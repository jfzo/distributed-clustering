function [snn_graph, knn] = compute_knn_snn(Z, k, distance_measure)
    % sparsified similarity matrix
    dmeasure = 'cosine';
    if nargin == 3
        dmeasure = distance_measure;
    end
    
    display('Starting k-near neighbor computation.');
    knn = snn_dd(Z, k, dmeasure);
    
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

function simval = snn_sim(snn_cell, i, j)
    % from the cell
    % returns the snn value

    simval = snn_cell{min(i,j)}(abs(i-j));
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

function [knn] = snn_dd(P, k, distance_measure)
    % returns a k+1 x N matrix with the indices 
    % of the smallest distances in each column
    % I(:, i ) denotes the k+1 elements with the smallest 
    % distances (including i )
    if nargin == 2
        distance_measure = 'cosine';
    end
 
       
    display('Computing pairwise distances.');
    N = size(P,1);
     
    if strcmp(distance_measure, 'cosine') == 1
        display('Using cosine distance measure.')
        [~, I] = pdist2(P, P, 'cosine', 'Smallest', k+1);
    elseif(strcmp(distance_measure, 'euclidean') == 1)
        display('Using euclidean distance measure.')
        [~, I] = pdist2(P, P, 'euclidean', 'Smallest', k+1);
    else
        display('Using cosine distance measure.')
        [~, I] = pdist2(P, P, 'cosine', 'Smallest', k+1);
    end
    
    knn = cell(N, 1);
    parfor i = 1:N
        nn_i = I(:, i);
        knn{i} = nn_i(2:end);%filters point i
    end
    display('k-near neighbor computed.');

end

