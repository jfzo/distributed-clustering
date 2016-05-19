function D = sqDistanceIn(X)
%Compute the square distances between data points in X 
% input:
% X: mXd data matrix, each row is a point
% output:
% D: mXm matrix, D(i,j) is the square distance between X(i,:) and X(i,:).
%    may have small negative values due to float computation

    A = dot(X,X,2);
    D = bsxfun(@plus, A, A') - 2 * (X * X');
	
    % if issparse(D) %needed if X is sparse and D should be full
    %     D=full(D);
    % end
end