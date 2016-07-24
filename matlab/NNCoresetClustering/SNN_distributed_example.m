% Z was generated as the CURE dataset and labels are its group tags
% Pwclass must be available!!

K = 70;
Nnodes = 4;
PCT_SAMPLE = 0.3;


G= random_graph_gen(Nnodes, 0.3);    
[N, ~] = size(Pwclass);
indn = get_partition('uniform', N, Nnodes, sum(G), 1, Pwclass(:,1:end-1));

KNN = cell(Nnodes, 1);
SNN = cell(Nnodes, 1);
parfor s=1:Nnodes
    localdata = Pwclass(indn==s,1:end-1);
    [KNN{s}, SNN{s}] = compute_knn_snn(localdata, K);
end

%Eps = 34;
%MinPts = 20;
range_Eps = 15:5:50;
range_MinPts = 15:5:30;

results_K = cell(length(range_Eps),length(range_MinPts));

for i=1:length(range_Eps)
    for j=1:length(range_MinPts)
        
        Eps = range_Eps(i);
        MinPts = range_MinPts(j);
        
        results_K{i,j} =  cell(5,1);
       
        display(sprintf('Distributed SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
        tic;
        [ results_K{i,j}{1}, results_K{i,j}{2}, results_K{i,j}{3}, results_K{i,j}{4} ] = Distributed_SNN(Pwclass, indn, Nnodes, SNN, K, Eps, MinPts, PCT_SAMPLE );
        %CORE_PTS_CT, CORE_CLST_CT, CT_DATA, CT_DATA_LBLS
        results_K{i,j}{5} = toc;
    end
end
save(sprintf('distributed_results_k%d.mat',K), 'K', 'Nnodes', 'range_Eps', 'range_MinPts', 'results_K', 'SNN')
        
% figure
% subplot(2,1,1)
% scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CT_DATA_LBLS,'o')
% title({['Centralized core-pts with original labels']});
% legend('off')
% subplot(2,1,2)
% scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CORE_CLST_CT,'o')
% title({['Centralized core-pts with identified labels']});
% legend('off')

