path('../DistributedCoresetAndPCA', path())
%%
clear;
P=importdata('../data/spiral.csv');
Pwclass = P;
P=P(:,1:2);

sigma_1 = 0.1;
sigma_2 = 0.1;
NOISE = mvnrnd([0,0], [sigma_1,0;0,sigma_2], round(size(P,1)));
%P = vertcat(P, NOISE);
P = P + NOISE;

%%
[labels, W] = SpectralClustering(P, 3, 0.95);
gscatter(P(:,1),P(:,2),labels(:,1))
%%