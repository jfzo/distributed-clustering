function [ CORE_PTS_CT, CORE_CLST_CT, CT_DATA, CT_DATA_LBLS ] = Distributed_SNN(DATASET, indn, Nnodes, SNN, K, Eps, MinPts, PCT_SAMPLE )
%DISTRIBUTED_SNN Summary of this function goes here
% Returns the resulting clusters of the transmitted core points from all
% sources to a centralized node.
% CORE_PTS_CT (vector): -1 if it is a noise point, 0 for non-core pts and 1 for core
% pts.
% CORE_CLST_CT (vector): 0 for noise and a value > 0 that denotes the
% cluster label identified in the centralized node for the point.
% CT_DATA (matrix): core-points transmitted from all nodes.
% CT_DATA_LBLS: Original labels (ground truth) for the core-points transmitted from all nodes.


DATA_LBLS = DATASET(:,end);
DATA = DATASET(:,1:end-1); 
%% Counting close points (in terms of SNN similarity) for each point ~ Density
%Eps = 34;

%Nnodes = 4;
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
%MinPts = 20;
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
%PCT_SAMPLE = 0.3;
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
%P = Pwclass(:,1:2);
%M_1 = P(indn==1,:);
% Perform SNN-clustering over the sampled core points.
N_sampled = 0;
for s=1:Nnodes
    N_sampled = N_sampled + length(SAMPLED_CORE_PTS{s});
end

CT_DATA = zeros(N_sampled, 2);
CT_DATA_LBLS = zeros(N_sampled, 1);

localdata = DATA(indn==1,:);
localdata_lbls = DATA_LBLS(indn==1,:);

CT_DATA(1:length(SAMPLED_CORE_PTS{1}),:) = localdata(SAMPLED_CORE_PTS{1},:);
CT_DATA_LBLS(1:length(SAMPLED_CORE_PTS{1}),:) = localdata_lbls(SAMPLED_CORE_PTS{1},:);

offset = length(SAMPLED_CORE_PTS{1}) + 1;
for s=2:Nnodes
    localdata = DATA(indn==s,:);
    localdata_lbls = DATA_LBLS(indn==s,:);
    CT_DATA(offset:offset+length(SAMPLED_CORE_PTS{s})-1,:) = localdata(SAMPLED_CORE_PTS{s},:);
    CT_DATA_LBLS(offset:offset+length(SAMPLED_CORE_PTS{s})-1,:) = localdata_lbls(SAMPLED_CORE_PTS{s},:);
    offset = offset + length(SAMPLED_CORE_PTS{s});
end

%% Apply SNN-clustering over CT_DATA

%K = 50;
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
%MinPts = 30;
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

% end of all
end

