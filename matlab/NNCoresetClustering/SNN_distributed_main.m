%% V2.0 - Experiments made from the DATA MATRIX
clear;
clc;
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng';'WSJ ';'AP
%']); % last two could not be processed.
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);
%textdatasets = cellstr(['SJMN';'FR  ']);
textdatasets = cellstr(['DOE ';'ZF  ';'20ng']);

range_Eps = horzcat([3 5 8 10], 15:5:50);
%range_Eps = 15:5:50;
range_MinPts = 5:5:30;

Nnodes = 4;
PCT_SAMPLE = 0.3;

%K = 15; % fixed

for ds=1:length(textdatasets)
    display(sprintf('Opening similarity matrix located at: ~/eswa-tfidf-data/%s_out.dat.dense', textdatasets{ds}));
    DATA = dlmread( sprintf('~/eswa-tfidf-data/%s_out.dat.dense', textdatasets{ds}) );
    LABELS = dlmread( sprintf('~/eswa-tfidf-data/%s_out.dat.labels', textdatasets{ds}) );
  
    Pwclass = horzcat(DATA, LABELS);
  
    G= random_graph_gen(Nnodes, 0.3);
    [N, ~] = size(DATA);
  
    indn = get_partition('uniform', N, Nnodes, sum(G), 1, Pwclass(:,1:end-1));
    for K=[50, 70, 90, 110]
        %KNN = cell(Nnodes, 1);
        SNN = cell(Nnodes, 1);
        parfor s=1:Nnodes
            localdata = Pwclass(indn==s,1:end-1);
            [SNN{s},~] = compute_knn_snn(localdata, K);
        end


        %Eps = 34;
        %MinPts = 20;
        results_K = cell(length(range_Eps),length(range_MinPts));

        for i=1:length(range_Eps)
            for j=1:length(range_MinPts)

                Eps = range_Eps(i);
                MinPts = range_MinPts(j);

                try
                    results_K{i,j} =  cell(5,1);

                    display(sprintf('Distributed SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
                    tic;
                    [ results_K{i,j}{1}, results_K{i,j}{2}, results_K{i,j}{3}, results_K{i,j}{4} ] = Distributed_SNN(Pwclass, indn, Nnodes, SNN, K, Eps, MinPts, PCT_SAMPLE );
                    %CORE_PTS_CT, CORE_CLST_CT, CT_DATA, CT_DATA_LBLS
                    results_K{i,j}{5} = toc;
                catch
                    results_K{i,j} = NaN;
                    display(sprintf('Could not end distributed SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
                end
            end
        end


        save(sprintf('tipster_results/distributed_%s_k%d.mat',textdatasets{ds},K), 'K', 'Nnodes', 'range_Eps', 'range_MinPts', 'results_K', 'SNN', '-v7.3')    
    end
end

%% To export the results into a format understood by the python script clustering_scores.py
clear;50, 70, 90, 
clc;
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);
%textdatasets = cellstr(['SJMN';'FR  ']);
textdatasets = cellstr(['DOE ';'ZF  ';'20ng']);


for ds=1:length(textdatasets)
    %for K=[50, 70, 90]
    for K=[90, 110]
        %load(sprintf('distributed_results_k%d.mat',K));
        load(sprintf('tipster_results/distributed_%s_k%d.mat',textdatasets{ds},K));

        for i=1:size(results_K,1)
            for j=1:size(results_K,2)

                if ~ iscell(results_K{i,j})
                    display(sprintf('Could not process results for dataset %s with K:%d Eps:%d MinPts:%d',textdatasets{ds},K,range_Eps(i), range_MinPts(j)));
                    continue;
                end

                A_LBLS = results_K{i,j}{2}(results_K{i,j}{1} ~= -1); % assigned labels
                T_LBLS = results_K{i,j}{4}(results_K{i,j}{1} ~= -1); % true labels
                display(sprintf('Writing results to tipster_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)) );
                csvwrite(sprintf('tipster_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat',textdatasets{ds}, K,range_Eps(i), range_MinPts(j)), horzcat(A_LBLS, T_LBLS))
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
end

%% COde for the cure dataset (toy example)
clear;
clc;

range_Eps = horzcat([3 5 8 10], 15:5:50);
range_MinPts = 5:5:30;

Nnodes = 8;
PCT_SAMPLE = 0.3;
%
DATA = dlmread('cure_data.csv');
LABELS = DATA(:,3);
DATA = DATA(:,1:2);

Pwclass = horzcat(DATA, LABELS);

G= random_graph_gen(Nnodes, 0.3);
[N, ~] = size(DATA);

indn = get_partition('uniform', N, Nnodes, sum(G), 1, Pwclass(:,1:end-1));


localdata = cell(Nnodes, 1);
for s=1:Nnodes
    localdata{s} = Pwclass(indn==s,1:end-1);
end

for K=[90, 110]
    SNN = cell(Nnodes, 1);
    
    parfor s=1:Nnodes
        %localdata = Pwclass(indn==s,1:end-1);
        [SNN{s},~] = compute_knn_snn(localdata{s}, K, 'euclidean');
    end
    
    results_K = cell(length(range_Eps),length(range_MinPts));
    
    for i=1:length(range_Eps)
        for j=1:length(range_MinPts)
            
            Eps = range_Eps(i);
            MinPts = range_MinPts(j);
            
            try
                results_K{i,j} =  cell(5,1);
                
                display(sprintf('Distributed SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
                tic;
                [ results_K{i,j}{1}, results_K{i,j}{2}, results_K{i,j}{3}, results_K{i,j}{4} ] = Distributed_SNN(Pwclass, indn, Nnodes, SNN, K, Eps, MinPts, PCT_SAMPLE );
                %CORE_PTS_CT, CORE_CLST_CT, CT_DATA, CT_DATA_LBLS
                results_K{i,j}{5} = toc;
            catch
                results_K{i,j} = NaN;
                display(sprintf('Could not end distributed SNN-clustering with parameters Eps:%d MinPts:%d (K:%d)\n',Eps, MinPts, K));
            end
        end
    end
    
    
    save(sprintf('cure_results/distributed_%s_k%d.mat','CureDs',K), 'K', 'Nnodes', 'range_Eps', 'range_MinPts', 'results_K', 'SNN', '-v7.3')
end

%% To export the results into a format understood by the python script clustering_scores.py

clear;
clc;

for K=[50, 70, 90, 110]
    load(sprintf('cure_results/distributed_%s_k%d.mat','CureDs',K));
    
    for i=1:size(results_K,1)
        for j=1:size(results_K,2)
            
            if ~ iscell(results_K{i,j})
                display(sprintf('Could not process results for dataset %s with K:%d Eps:%d MinPts:%d','CureDs',K,range_Eps(i), range_MinPts(j)));
                continue;
            end
            
            A_LBLS = results_K{i,j}{2}(results_K{i,j}{1} ~= -1); % assigned labels
            T_LBLS = results_K{i,j}{4}(results_K{i,j}{1} ~= -1); % true labels
            display(sprintf('Writing results to cure_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat','CureDs', K,range_Eps(i), range_MinPts(j)) );
            csvwrite(sprintf('cure_results/figs/distributed_%s_k%d_eps%d_minpts%d.dat','CureDs', K,range_Eps(i), range_MinPts(j)), horzcat(A_LBLS, T_LBLS))
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
        load(sprintf('tipster_results/distributed_%s_k%d.mat',textdatasets{ds},K));

        for i=1:size(results_K,1)
            for j=1:size(results_K,2)

                if ~ iscell(results_K{i,j})
                    display(sprintf('Could not process results for dataset %s with K:%d Eps:%d MinPts:%d',textdatasets{ds},K,range_Eps(i), range_MinPts(j)));
                    continue;
                end
                etimes{c,1} = [etimes{c,1}; results_K{i,j}{5}];

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
R = ETDATA('20ng')
x = [R{1,1};R{2,1};R{3,1};R{4,1}];
g = [ones(size(R{1,1}));2*ones(size(R{2,1}));3*ones(size(R{3,1}));4*ones(size(R{4,1}))];
boxplot(x,g)
set(gca,'XTickLabel',{'K=50','K=70','K=90','K=110'})
ylabel('time secs.')
ylabel('time [secs]')
