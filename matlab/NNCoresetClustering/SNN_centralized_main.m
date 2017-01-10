%% V1.0 - Experiments made from the raw data vectors
K = 70; % fixed
DATA = Pwclass(:,1:2);
[SNN_K] = compute_knn_snn(DATA, K);

range_Eps = horzcat([3 5 8 10], 15:5:50);
range_MinPts = 15:5:30;

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
save(sprintf('global_results_k%d-1.mat',K), 'K', 'range_Eps', 'range_MinPts', 'results_K', 'SNN_K')

%% V2.0 - Experiments made from the similarity matrix
clear;
clc
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng';'WSJ ';'AP  ']);
textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);

%K = 70; % fixed
range_Eps = horzcat([3 5 8 10], 15:5:50);
range_MinPts = 15:5:30;

for ds=1:length(textdatasets)
    display(sprintf('Opening similarity matrix located at: ~/eswa-tfidf-data/%s_out.dat_sim.csv', textdatasets{ds}));
    DATA = dlmread( sprintf('~/eswa-tfidf-data/%s_out.dat_sim.csv', textdatasets{ds}) );
    display(sprintf('DATA size %d %d',size(DATA,1),size(DATA,2)))
    % DATA corresponds to a similarity matrix already computed.
    
    for K=[90, 110]
        [SNN_K] = compute_knn_snn_from_similarity(DATA, K);


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
        save(sprintf('tipster_results/centralized_%s_k%d-1.mat',textdatasets{ds},K), 'K', 'range_Eps', 'range_MinPts', 'results_K', 'SNN_K')
    end
end
%% Plotting and storing the figures of the obtained results.
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

%% To export the results into a format understood by the python script clustering_scores.py
clear;
clc;
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng';'WSJ ';'AP  ']);
textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);


for ds=1:length(textdatasets)
    %for K=[50, 70, 90]
    labels = dlmread( sprintf('~/eswa-tfidf-data/%s_out.dat.labels', textdatasets{ds}) );
    for K=[90, 110]
        %load(sprintf('global_results_k%d.mat',K));
        load(sprintf('tipster_results/centralized_%s_k%d-1.mat',textdatasets{ds},K));

        for i=1:size(results_K,1)
            for j=1:size(results_K,2)
                A_LBLS = results_K{i,j}{2}(results_K{i,j}{1} ~= -1); % assigned labels
                T_LBLS = labels(results_K{i,j}{1} ~= -1); % true labels
                display(sprintf('Writing results to tipster_results/figs/centralized_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)) );
                csvwrite(sprintf('tipster_results/figs/centralized_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)), horzcat(A_LBLS, T_LBLS));
            end
        end
    end
end

%% To perform an analysis of the elapsed times
clear;
clc;
textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);
%textdatasets = cellstr(['SJMN';'FR  ']);
%textdatasets = cellstr(['DOE ';'ZF  ';'20ng']);
%textdatasets = cellstr(['20ng']);

ETDATA = containers.Map([textdatasets], cell(length(textdatasets),1));

for ds=1:length(textdatasets)
    %for K=[50, 70, 90]
    etimes = cell(4, 1); % One for each value of K
    c = 1;
    for K=[50, 70, 90, 110]
        %load(sprintf('distributed_results_k%d.mat',K));
        load(sprintf('tipster_results/centralized_%s_k%d.mat',textdatasets{ds},K));

        for i=1:size(results_K,1)
            for j=1:size(results_K,2)

                if ~ iscell(results_K{i,j})
                    display(sprintf('Could not process results for dataset %s with K:%d Eps:%d MinPts:%d',textdatasets{ds},K,range_Eps(i), range_MinPts(j)));
                    continue;
                end
                etimes{c,1} = [etimes{c,1}; results_K{i,j}{3}];

                %display(sprintf('Writing results to tipster_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)) );
                %csvwrite(sprintf('tipster_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)), horzcat(A_LBLS, T_LBLS))
            end
        end
        %subplot(2,2,c);
        %boxplot(etimes{c,1});
        c = c + 1;
    end
    ETDATA(textdatasets{ds}) = etimes;
        
    % Generate the boxplot for each value of K
    % figure
    % subplot(2,1,1)
    % scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CT_DATA_LBLS,'o')
    % title({['Centralized core-pts with original labels']});
    % legend('off')
    % subplot(2,1,2)
    % scatter(CT_DATA(:,1), CT_DATA(:,2), 5, CORE_CLST_CT,'o')
    % title({['Centralized core-pts with identified labels']});
    % legend('off')
end

%'20ng'    'DOE'    'FR'    'SJMN'    'ZF'
R = ETDATA('ZF')
x = [R{1,1};R{2,1};R{3,1};R{4,1}];
g = [ones(size(R{1,1}));2*ones(size(R{2,1}));3*ones(size(R{3,1}));4*ones(size(R{4,1}))];
boxplot(x,g)
set(gca,'XTickLabel',{'K=50','K=70','K=90','K=110'})
ylabel('time secs.')
ylabel('time [secs]')