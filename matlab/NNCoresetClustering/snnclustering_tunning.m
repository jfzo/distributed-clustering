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
        results_K{i,j}{3} = toc;
    end
end
save(sprintf('global_results_k%d.mat',K), 'K', 'range_Eps', 'range_MinPts', 'results_K', 'SNN_K')
