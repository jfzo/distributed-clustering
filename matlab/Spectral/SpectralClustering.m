function [ labels, weights ] = SpectralClustering( P, K, sigma )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    A = pdist2(P,P);
    A = exp(-(A.^2)/( 2 * sigma^2 ) );
    for i=1:size(A,1)   
        A(i,i) = 0;
    end

    D = sum(A, 2);
    D_A = diag((D.^-1).^(1/2));
    L = D_A * A * D_A;
    [X, eVal] = eigs(L, K);

    Y = diag(1./(sum(X.^2,2).^(1/2)) ) * X;
    [centers]=lloyd_kmeans(K, Y);

    dims = size(Y);
    labels = zeros(dims(1), 1);
    weights =  zeros(dims(1), 1);

    for i=1:dims(1)
        min_d = inf;
        min_c = -1;
        for c=1:size(centers,1)
            curr_d = sum((Y(i,:)-centers(c,:)).^2);
            if curr_d < min_d
                min_d = curr_d;
                min_c = c;
            end
        end
        labels(i,1) = min_c;
        weights(i,1) = min_d;
    end

end

