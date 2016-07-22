% Z was generated as the CURE dataset and labels are its group tags
% Pwclass must be available!!
K = 50;
Nnodes = 4;
Eps = 34;
MinPts = 20;
PCT_SAMPLE = 0.3;

G= random_graph_gen(Nnodes, 0.3);    
[N, ~] = size(DATASET);
indn = get_partition('uniform', N, Nnodes, sum(G), 1, Pwclass(:,1:2));

KNN = cell(Nnodes, 1);
SNN = cell(Nnodes, 1);
parfor s=1:Nnodes
    localdata = P(indn==s,:);
    [KNN{s}, SNN{s}] = compute_knn_snn(localdata, K);
end

[ CORE_PTS_CT, CORE_CLST_CT, CT_DATA, CT_DATA_LBLS ] = Distributed_SNN(Pwclass, indn, Nnodes, SNN, K, Eps, MinPts, PCT_SAMPLE );

figure
subplot(2,1,1)
scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CT_DATA_LBLS,'o')
title({['Centralized core-pts with original labels']});
legend('off')
subplot(2,1,2)
scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CORE_CLST_CT,'o')
title({['Centralized core-pts with identified labels']});
legend('off')

