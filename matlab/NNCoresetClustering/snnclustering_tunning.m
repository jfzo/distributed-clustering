%%
K = 90; % fixed
DATA = Pwclass(:,1:2);
[~, SNN_K] = compute_knn_snn(DATA, K);

<<<<<<< HEAD
range_Eps = 15:5:60;
range_MinPts = 15:5:45;
=======
range_Eps = 15:5:50;
range_MinPts = 15:5:30;
>>>>>>> db61ceab6fb4ea1096990d3309fde3bc63079ab0

results_K = cell(length(range_Eps),length(range_MinPts));

for i=1:length(range_Eps)
    for j=1:length(range_MinPts)
        tic
        Eps = range_Eps(i);
        MinPts = range_MinPts(j);
        
        results_K{i,j} =  cell(3,1);
       
        display(sprintf('SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
        [results_K{i,j}{1}, results_K{i,j}{2}] =  SNNClustering_from_snnsim(SNN_K, Eps, MinPts);
        %CORE_PTS_CT, CORE_CLST_CT
        results_K{i,j}{3} = toc;
    end
end
save(sprintf('global_results_k%d.mat',K), 'K', 'range_Eps', 'range_MinPts', 'results_K', 'SNN_K')

%%
DATA = Pwclass(:,1:2);
DATA_LBLS = Pwclass(:,3);

load('global_results_k90.mat')
for i=1:size(results_K,1)
    for j=1:size(results_K,2)

        CORE_PTS_CT = results_K{i,j}{1};
        CORE_CLST_CT = results_K{i,j}{2};

        fig = figure;
        subplot(2,1,1)
        scatter(DATA(:,1), DATA(:,2), 5, DATA_LBLS,'o')
        title({['Centralized core-pts with original labels'];['Eps:',num2str(range_Eps(i)),'  MinPts:',num2str(range_MinPts(j))]});
        legend('off')
        subplot(2,1,2)
        scatter(DATA(:,1), DATA(:,2), 5, CORE_CLST_CT,'o')
        title({['Centralized core-pts with identified labels']});
        legend('off')
        
        saveas(fig, sprintf('figs/K90/global_clustering_result_K%d_eps%d_minpts%d.png',K, range_Eps(i), range_MinPts(j) ));
        close(fig)
    end
end
