function [centers, labels, weights ] = SpectralClustering( P, K, sigma, with_distance )
%SpectralClustering Computes the spectral clustering algorithm
%   Accepts 3 or 4 parameters where
% 1st denotes the data or the distance matrix (the fourth parameter must be
% set to true in this case)
% 2nd denotes the number of clusters to identify
% 3rd denotes the $\sigma$ parameter of the smoothing operation: exp(-(A.^2)/( 2 * sigma^2 ) )
    
    if nargin == 4 % (with_distance is set) Then P is a distance matrix...
        A = P;
    else
        A = pdist2(P,P);
    end
    
    A = exp(-(A.^2)/( 2 * sigma^2 ) );
    for i=1:size(A,1)   
        A(i,i) = 0;
    end

    D = sum(A, 2);
    D_A = diag((D.^-1).^(1/2));
    L = D_A * A * D_A;
    
    [X, eVal] = eigs(L, K); % X \in \R^{n\times K}
    
    %opts.tol = 1e-3; 
    %[X, eVal] = eigs(L, K, 'lr', opts); 


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

