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

%% computing SNN density
Eps = 35;
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
