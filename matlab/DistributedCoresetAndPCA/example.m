%Script to show one example run for Distributed PCA analysis of 
%k-means Clustering algorithm
%Dataset considered is Pendigits of size: N=10992, dim=16 distributed using a
%weighted partitioning method, over a random graph network with n=9,p=0.3.
%We compute clusters(k=10, t=5000) using distributed coreset and Lloyd's kmeans 
%algorithms, of the data reduced to dimension t_vector=[14].

	%Load data
	%P=importdata('pendigits.mat');
    %P=importdata('./data/noisy_circles.csv');
    %P=importdata('./data/noisy_moons.csv');
    %P=importdata('./data/spiral.csv');
    %Pwclass = P        
    
%%
path('./DistributedCoresetAndPCA', path())
%%
%     clear;
%     P=importdata('vary-density.csv');
%     Pwclass = P;
%     Z = zeros(size(Pwclass));
%     Z(1:50,:) = Pwclass(Pwclass(:,3)==3,:);
%     Z(51:100,1) = Pwclass(Pwclass(:,3)==2,1)-0.2;
%     Z(51:100,2) = Pwclass(Pwclass(:,3)==2,2)-0.2;
%     Z(51:100,3) = zeros(50,1)+1;
%     Z(101:end,:) = Pwclass(Pwclass(:,3)==1,:);
%     gscatter(Z(:,1),Z(:,2),Z(:,3))
%     Pwclass = Z;
%     P = Z(:,1:2);
%% Spiral (2D data)
clear;
Pwclass=importdata('spiral.csv');
gscatter(Pwclass(:,1),Pwclass(:,2),Pwclasssss(:,3))
% adding noise
P = Pwclass(:,1:2);
sigma_1 = 0.1;
sigma_2 = 0.1;
NOISE = mvnrnd([0,0], [sigma_1,0;0,sigma_2], round(size(P,1)));
%P = vertcat(P, NOISE);
P = P + NOISE;

gscatter(P(:,1),P(:,2),Pwclass(:,3))
%%
clear;
clc;
Pwclass=importdata('noisy_circles.csv'); % Use parameter $\sigma=0.1$
%gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
% adding noise
P = Pwclass(:,1:2);
%% Coreset routine    
K = length(unique(Pwclass(:,3))); % it can be manually set also!    

%'Random'graph generation with n=9, p=0.3
Nnodes = 3;
G= random_graph_gen(Nnodes, 0.3);    
fprintf('generated random graph\n');

%partitioning data into 9 local sets using 'weighted' partition method
[N, dim] = size(P);
indn=get_partition('weighted', N, Nnodes, sum(G), 1, P);

%Distributed PCA of the data with t_vector = [14]
%%proj_vector = distributed_pca(P, [14], 9, indn);
%%lowDim_P = P*proj_vector{1};
lowDim_P = P;

global LOC_CORESET_CENTERS
LOC_CORESET_CENTERS = zeros(Nnodes*K, 2);
%Distributed_coreset construction and lloyd's k-means impementation
%for the PCA data with k=10, t=10% of the size of the data
[S,w] = distributed_coreset(lowDim_P, indn, Nnodes, K, floor(0.2*N) );
%% Visualzation of the coreset and its labels set with kmeans
[centers_coreset]=lloyd_kmeans(K, S, w);

dims = size(S);
labeledS = zeros(dims(1), dims(2) + 1);
labeledS(:,1:2) = S;

for i=1:dims(1)
    min_d = inf;
    min_c = -1;
    for c=1:size(centers_coreset,1)
        curr_d = sum((S(i,:)-centers_coreset(c,:)).^2);
        if curr_d < min_d
            min_d = curr_d;
            min_c = c;
        end
    end
    labeledS(i,3) = min_c;
end

gscatter(labeledS(:,1),labeledS(:,2),labeledS(:,3))

%% Building the final clusters with spectral clustering 
clc;
[centers, labelsSC, W] = SpectralClustering(S, K, 0.77);
gscatter(S(:,1),S(:,2),labelsSC)





%% Finding the closest center to each coreset point
[centers_entire]=lloyd_kmeans(K, P); % It has the centerd of the whole data

local_centers=zeros(Nnodes*K, 2);
j=1;
for i=1:Nnodes
    local_centers(j,:) = LOC_CORESET_CENTERS{i}(1,:);
    j=j+1;
    local_centers(j,:) = LOC_CORESET_CENTERS{i}(2,:);
    j=j+1;
end

% Coreset points with labels added in the last column by
%  finding the nearest centroid in the coreset.
% Note that each center in the coreset is found by performing K-means
%  over the coreset.
dims = size(S);
labeledS = zeros(dims(1), dims(2) + 1);
labeledS(:,1:2) = S;

for i=1:dims(1)
    min_d = inf;
    min_c = -1;
    for c=1:size(centers_coreset,1)
        curr_d = sum((S(i,:)-centers_coreset(c,:)).^2);
        if curr_d < min_d
            min_d = curr_d;
            min_c = c;
        end
    end
    labeledS(i,3) = min_c;
end

figure1 = figure('PaperSize',[20.98404194812 29.67743169791]);
axes1 = axes('Parent',figure1);
hold(axes1,'all');

%line(local_centers(:,1),local_centers(:,2),'Parent',axes1,'MarkerFaceColor',[0 1 0],'Marker','diamond','LineStyle','none',...
%'Color',[1 0.5 0.5],'MarkerSize',10,'DisplayName','Local node centers');

%gscatter(P(:,1), P(:,2), ones(size(P,1)), 'm')
gscatter(labeledS(:,1),labeledS(:,2),labeledS(:,3))

%% Visualization of centroids, coreset and original data with their labels.
local_centers=zeros(Nnodes*K, 2);
j=1;
for i=1:Nnodes
    local_centers(j,:) = LOC_CORESET_CENTERS{i}(1,:);
    j=j+1;
    local_centers(j,:) = LOC_CORESET_CENTERS{i}(2,:);
    j=j+1;
end

figure1 = figure('PaperSize',[20.98404194812 29.67743169791]);
axes1 = axes('Parent',figure1);
hold(axes1,'all');

line(local_centers(:,1),local_centers(:,2),'Parent',axes1,'MarkerFaceColor',[0 1 0],'Marker','diamond','LineStyle','none',...
'Color',[1 0.5 0.5],'MarkerSize',10,'DisplayName','Local node centers');


plot(S(:,1), S(:,2),'Parent',axes1,'MarkerFaceColor',...
[0.749019622802734 0.749019622802734 0],'MarkerSize',8,'Marker','+','LineStyle','none',...
'DisplayName','Coreset data');

plot(centers_coreset(:,1), centers_coreset(:,2),'Parent',axes1,'MarkerSize',10,'Marker',...
'x','LineWidth',4,'LineStyle','none','Color',[0.87058824300766 0.490196079015732 0],...
'DisplayName','Coreset centroids');

plot(centers_entire(:,1), centers_entire(:,2),'Parent',axes1,'MarkerSize',8,'Marker',...
'*','LineWidth',4,'LineStyle','none','Color',[0 0 0],...
'DisplayName','Real centroids');

%gscatter(Z(:,1),Z(:,2),Z(:,3));

line(P(Pwclass(:,3)==1,1),P(Pwclass(:,3)==1,2),'Parent',axes1,'Marker','square','LineStyle','none',...
'Color',[1 0 0],'DisplayName','Orig.Data, class A');

line(P(Pwclass(:,3)==2,1),P(Pwclass(:,3)==2,2),'Parent',axes1,'MarkerSize',8,'Marker','v',...
'LineStyle','none','Color',[0 1 1],'DisplayName','Orig.Data, class B');


line(P(Pwclass(:,3)==3,1),P(Pwclass(:,3)==3,2),'Parent',axes1,'MarkerSize',8,'Marker','o',...
'LineStyle','none','Color',[0 0 1],'DisplayName','Orig.Data, class C');
legend(axes1,'show');
%% Part that computes the overall cost.
    
centers_dim = centers_coreset*proj_vector{1}';
y = sqDistance(centers_dim, P);
min_y= min(y);
kMeansCoresetCost = sum(min_y);
fprintf('kmeans cost at dim 14=%f\n',kMeansCoresetCost);
CommunicationCost = size(S,1)*size(S,2)*sum(G(:));

%PCA communication cost
PCA_comm_cost = (1+ dim)*9 + (14+14*dim)*9;
CommunicationCost = CommunicationCost + sum(G(:)) * PCA_comm_cost;
fprintf('coreset_comm_cost at dim 14 =%f\n', CommunicationCost);
    
    
    