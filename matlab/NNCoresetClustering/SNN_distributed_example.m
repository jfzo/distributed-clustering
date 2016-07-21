% Z was generated as the CURE dataset and labels are its group tags
P = Z;
Pwclass = horzcat(Z, labels);


K = length(unique(Pwclass(:,3))); % it can be manually set also!    

%'Random'graph generation with n=9, p=0.3
Nnodes = 4;
G= random_graph_gen(Nnodes, 0.3);    
fprintf('generated random graph\n');

%partitioning data into 9 local sets using 'weighted' partition method
[N, dim] = size(P);
indn=get_partition('uniform', N, Nnodes, sum(G), 1, P);

%% computing top-k near neighbors and shared near neighbors
% Consider node 's'. Then SNN{s}{i} is an array that contains for point i, 
% the number of snn's between item i and those items having an ID > i . 
% THUS, SNN{s}{i}(1) has the value of snn's between i and (i+1) .

K = 50;
KNN = cell(Nnodes, 1);
SNN = cell(Nnodes, 1);
parfor s=1:Nnodes
    localdata = P(indn==s,:);
    [KNN{s}, SNN{s}] = compute_knn_snn(localdata, K);
end

%% Counting close points (in terms of SNN similarity) for each point ~ Density
Eps = 34;

Nnodes = 4;
DST = cell(Nnodes, 1);
parfor s=1:Nnodes 
    DST{s} = zeros(length(SNN{s}) + 1, 1); %array to store density
    for i=1:length(SNN{s})
        dense_ng = find(SNN{s}{i} > Eps);
        DST{s}(i) = DST{s}(i) + length(dense_ng);
        for dense_item=dense_ng %updates the counters of the neighb.
            DST{s}(i+dense_item) = DST{s}(i+dense_item) + 1
        end
    end
end

%% computing CORE points
MinPts = 20;
% snn similarity function: e.g. snn_sim(SNN{1},7,1)
snn_sim = @(SNN_info, i, j) SNN_info{min(i,j)}(abs(i-j));

%DST{1} > MinPts
CORE_PTS = cell(Nnodes, 1);
CORE_CLST = cell(Nnodes, 1);
parfor s=1:Nnodes
    CORE_PTS{s} = DST{s} > MinPts;
    CORE_CLST{s} = zeros(length(DST{s}),1);
    core_points = find(DST{s} > MinPts);
  
    clst_id = 0;
    for i=1:size(core_points, 1)-1 %identifying clusters
        if CORE_CLST{s}(core_points(i)) == 0
            clst_id = clst_id + 1;
            CORE_CLST{s}(core_points(i)) = clst_id;
        end
        for j=i+1:size(core_points, 1)
            if snn_sim(SNN{s},core_points(i),core_points(j)) >= Eps
                CORE_CLST{s}(core_points(j)) = CORE_CLST{s}(core_points(i));
            end
        end
    end
end
% remember to filter the near ones.
%% sampling core-points from each node
PCT_SAMPLE = 0.3;
SAMPLED_CORE_PTS = cell(Nnodes, 1);
parfor s=1:Nnodes
    %find(CORE_PTS{s} == 1)
    PTS_s_W = zeros(length(CORE_PTS{s}), 1);
    N_s = length(find(CORE_CLST{s} ~= 0));
    
    clst_s = unique(CORE_CLST{s});
    for c=1:length(clst_s)
        if clst_s(c) == 0
            continue
        end 
        %display(sprintf('[%d] Valor de c=%d \n', s, clst_s(c)));
        N_c = length(find(CORE_CLST{s} == clst_s(c)));
        %display(sprintf('Valor de N_c=%d \n', N_c));
        w_c = ((1 - ( N_c/N_s ) )/2) / N_c;
        PTS_s_W(CORE_CLST{s} == clst_s(c)) = w_c;
    end
    %display( sum(PTS_s_W) );
    SAMPLED_CORE_PTS{s} = randsample(length(CORE_CLST{s}), round(PCT_SAMPLE*N_s) ,true, PTS_s_W);
end

%% Centralizing the transmitted core-points into a single dataset 
P = Pwclass(:,1:2);
%M_1 = P(indn==1,:);
% Perform SNN-clustering over the sampled core points.
N_sampled = 0;
for s=1:Nnodes
    N_sampled = N_sampled + length(SAMPLED_CORE_PTS{s});
end

CT_DATA = zeros(N_sampled, 2);

localdata = P(indn==1,:);
CT_DATA(1:length(SAMPLED_CORE_PTS{1}),:) = localdata(SAMPLED_CORE_PTS{1},:);
offset = length(SAMPLED_CORE_PTS{1}) + 1;
for s=2:Nnodes    
    localdata = P(indn==s,:);
    CT_DATA(offset:offset+length(SAMPLED_CORE_PTS{s})-1,:) = localdata(SAMPLED_CORE_PTS{s},:);
    offset = offset + length(SAMPLED_CORE_PTS{s});    
end

%% Apply SNN-clustering over CT_DATA

K = 50;
[KNN_CT, SNN_CT] = compute_knn_snn(CT_DATA, K);

% Counting close points (in terms of SNN similarity) for each point ~ Density
Eps = 15;
DST_CT = zeros(length(SNN_CT) + 1, 1); %array to store density
for i=1:length(SNN_CT)
    dense_ng = find(SNN_CT{i} > Eps);
    DST_CT(i) = DST_CT(i) + length(dense_ng);
    for dense_item=1:length(dense_ng) %updates the counters of the neighb.
        DST_CT(i+dense_ng(dense_item)) = DST_CT(i+dense_ng(dense_item)) + 1;
    end
end

% Identifying CORE points
MinPts = 30;
% snn similarity function: e.g. snn_sim(SNN{1},7,1)
snn_sim = @(SNN_info, i, j) SNN_info{min(i,j)}(abs(i-j));
CORE_PTS_CT = DST_CT > MinPts;
CORE_CLST_CT = zeros(length(DST_CT),1);
core_points = find(DST_CT > MinPts);
clst_id = 0;
for i=1:size(core_points, 1)-1 %identifying clusters
    if CORE_CLST_CT(core_points(i)) == 0
        clst_id = clst_id + 1;
        CORE_CLST_CT(core_points(i)) = clst_id;
    end
    for j=i+1:size(core_points, 1)
        if snn_sim(SNN_CT,core_points(i),core_points(j)) >= Eps
            CORE_CLST_CT(core_points(j)) = CORE_CLST_CT(core_points(i));
        end
    end
end

% Discarding Noise points (marked with -1 in its entry in CORE_PTS_CT)
non_core = find(CORE_PTS_CT == 0 );
core = find(CORE_PTS_CT == 1);
core_clst_id = CORE_CLST_CT; % copy made to enable parallel loop
for i=1:length(non_core)
    % find nearest core point
    nearest_sim = 0;
    nearest_core = -1;
    for j=1:length(core)
        j_sim = snn_sim(SNN_CT, non_core(i), core(j));
        if j_sim > nearest_sim
            nearest_sim = j_sim;
            nearest_core = core(j);
        end
    end
    % if the snn similarity is lower than Eps => discard the point    
    if nearest_sim < Eps
        CORE_PTS_CT(non_core(i)) = -1;
    else % otherwise, label the point with the nearest core point label.
        CORE_CLST_CT(non_core(i)) = core_clst_id(nearest_core);
    end
end


% Plotting core points with their identified labels
scatter(CT_DATA(CORE_PTS_CT,1), CT_DATA(CORE_PTS_CT,2), 5, CORE_CLST_CT(CORE_PTS_CT),'o')
% Plotting non-core/non-noise points with their identified labels
non_core = find(CORE_PTS_CT ~= 0 );
if isempty(non_core) > 0
    scatter(CT_DATA(non_core,1), CT_DATA(non_core,2), 5, CORE_CLST_CT(non_core),'o')
end

% Export core and non-core/non-noise points with their labels 

%[A,B] = SNNClustering( CT_DATA, K, Eps, MinPts);
%% Plotting for 4 nodes

figure
for s=1:Nnodes
    subplot(2,2,s)
    Dn = Pwclass(indn == s,:);hold on;
    scatter(Dn(:,1),Dn(:,2), 2, '.k');
    % %gscatter(Dn(CORE_PTS{s},1), Dn(CORE_PTS{s},2), CORE_CLST{s}(CORE_PTS{s}))
    scatter(Dn(CORE_PTS{s},1), Dn(CORE_PTS{s},2), 5, CORE_CLST{s}(CORE_PTS{s}),'o')
    title({['Core-pts in Node ',num2str(s)];[' (',num2str(100*sum(CORE_PTS{s})/size(Dn,1)),'%)']});
    legend('off')
end



figure
for s=1:Nnodes
    subplot(2,2,s)
    N = length(KNN{s});
    knnG_s = zeros(length(KNN{s}));
    for j=1:N-1
        for nn=KNN{s}{j}
            knnG_s(j,nn)= 1;
            knnG_s(nn,j)= 1;
        end
    end

    Gs=graph(knnG_s,'upper');
    p = plot(Gs);
    p.NodeColor = 'red';
    p.LineWidth = 0.1;
    Gs.Nodes.NodeColors = degree(Gs);
    p.NodeCData = Gs.Nodes.NodeColors;
    colorbar
    title({['Node ',num2str(s)];['']});
end
