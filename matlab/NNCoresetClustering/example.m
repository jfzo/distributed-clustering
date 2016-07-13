%%
% Example usage of the nn distributed coreset.

%% Spiral (2D data)
clear;
clc;
Pwclass=importdata('noisy_circles.csv');
gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
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
gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
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

[S,w] = nnd_coreset(lowDim_P, indn, Nnodes, K, floor(0.2*N) );

[centers, labels, W] = SpectralClustering(S, K, 0.1);

gscatter(S(:,1),S(:,2),labels)
    
%% Finding the closest center to each coreset point

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