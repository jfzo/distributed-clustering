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
range_Eps = horzcat([3 5 8 10], 15:5:50);
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
        
%% Plotting and storing the figures of the obtained results.
load('distributed_results_k90.mat')

for i=1:size(results_K,1)
    for j=1:size(results_K,2)

        DATA = results_K{i,j}{3};
        DATA_LBLS = results_K{i,j}{4};
        
        CORE_PTS_CT = results_K{i,j}{1};
        CORE_CLST_CT = results_K{i,j}{2};

        fig = figure;
        subplot(2,1,1)
        scatter(DATA(:,1), DATA(:,2), 5, DATA_LBLS,'o')
        title({['Core-pts with original labels'];['Eps:',num2str(range_Eps(i)),'  MinPts:',num2str(range_MinPts(j))]});
        legend('off')
        subplot(2,1,2)
        scatter(DATA(:,1), DATA(:,2), 5, CORE_CLST_CT,'o')
        title({['Core-pts with identified labels']});
        legend('off')
        
        saveas(fig, sprintf('results/distribuido/K90/distributed_clustering_result_K%d_eps%d_minpts%d.png',K, range_Eps(i), range_MinPts(j) ));
        close(fig)
    end
end

%% To export the results into a format understood by the python script clustering_scores.py
clc;
for K=[50, 70, 90]
    load(sprintf('distributed_results_k%d.mat',K));

    for i=1:size(results_K,1)
        for j=1:size(results_K,2)
            A_LBLS = results_K{i,j}{2}(results_K{i,j}{1} ~= -1); % assigned labels
            T_LBLS = results_K{i,j}{4}(results_K{i,j}{1} ~= -1); % true labels
            display(sprintf('Writing results to results/distribuido/K%d/distributed_clustering_result_K%d_eps%d_minpts%d.dat',K, K, range_Eps(i), range_MinPts(j)) );
            csvwrite(sprintf('results/distribuido/K%d/distributed_clustering_result_K%d_eps%d_minpts%d.dat',K, K,range_Eps(i), range_MinPts(j)), horzcat(A_LBLS, T_LBLS))
        end
    end
end
% figure
% subplot(2,1,1)
% scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CT_DATA_LBLS,'o')
% title({['Centralized core-pts with original labels']});
% legend('off')
% subplot(2,1,2)
% scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CORE_CLST_CT,'o')
% title({['Centralized core-pts with identified labels']});
% legend('off')

