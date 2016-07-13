% parameters
% Eps y knn
Eps = 20;
knn = 20;
%%%%
clear;
clc;
Pwclass=importdata('noisy_circles.csv'); % Use parameter $\sigma=0.1$
gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
% adding noise
P = Pwclass(:,1:2);
%% Building and sparsifying the distance matrix
A = pdist2(P,P);

for row=1:size(A,1)
    [Sval, Sind] = sort(A(row,:),'ascend');
    A(row, Sind(knn+2:end)) = 0; % consider self-similarity
end
%% SNN graph
B = zeros(size(A));
for row_i=1:(size(A,1)-1)
    for row_j=(row_i+1):size(A,1)
        B(row_i, row_j) = sum((A(row_i,:) > 0) & (A(row_j,:) > 0));
    end
end
display('Ok!');

%% CURE DATASET GENERATOR
clc;
% The big circle
mu = [0 -0.5];
Sigma = [5 0; 0 1]; R = chol(Sigma);
n1 = 3000;
z1 = repmat(mu,n1,1) + randn(n1,2)*R;
l1 = zeros(n1,1) + 1;

%the 1st elipse and its bridge
mu = [-2 6];
Sigma = [3 0; 0 0.05]; R = chol(Sigma);
n2 = 1000;
z2 = repmat(mu,n2,1) + randn(n2,2)*R;
l2 = zeros(n2,1) + 2;


mu = [1.0 5.9];
Sigma = [20 0; 0 0.01]; R = chol(Sigma);
n6 = 100;
z6 = repmat(mu,n6,1) + rand(n6,2)*R;
l6 = zeros(n6,1) + 2;

%the 2nd elipse
mu = [10 6];
Sigma = [3 0; 0 0.05]; R = chol(Sigma);
n3 = 1000;
z3 = repmat(mu,n3,1) + randn(n3,2)*R;
l3 = zeros(n3,1) + 3;

%the upper small circle
mu = [15 1];
Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
n4 = 1000;
z4 = repmat(mu,n4,1) + randn(n4,2)*R;
l4 = zeros(n4,1) + 4;

%the lower small circle
mu = [15 -2];
Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
n5 = 1000;
z5 = repmat(mu,n5,1) + randn(n5,2)*R;
l5 = zeros(n5,1) + 5;

% background noise
mu = [-8 -5];
Sigma = [700 0; 0 150]; R = chol(Sigma);
n7 = 1000;
z7 = repmat(mu,n7,1) + rand(n7,2)*R;
l7 = zeros(n7,1) + 7;

% merging 
Z = vertcat(z1, z2, z3, z4, z5,z6, z7);
labels = vertcat(l1,l2,l3,l4,l5,l6,l7);
gscatter(Z(:,1),Z(:,2), labels )
