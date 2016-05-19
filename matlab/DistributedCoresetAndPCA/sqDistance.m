function D = sqDistance(X, Y)
%Compute the square distances between data points in X and Y
% input:
% X: mXd data matrix, each row is a point
% Y: nXd data matrix, each row is a point
% output:
% D: mXn matrix, D(i,j) is the square distance between X(i,:) and Y(j,:).
%    may have small negative values due to float computation

    D = bsxfun(@plus, dot(X,X,2), dot(Y,Y,2)') - 2 * (X * Y');
end