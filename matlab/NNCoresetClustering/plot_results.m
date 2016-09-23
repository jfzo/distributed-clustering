%% Centralized
% ARI 4
% AMI 5
% VMeasure 6
clear;
clc;
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng';'WSJ ';'AP  ']);
textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);

for ds=1:length(textdatasets)
    for K=[90, 110]
        loaded_results = csvread(sprintf('tipster_results/figs/centralized_%s_k%d_results_ARI_AMI_VM',textdatasets{ds},K));
        %loaded_results = csvread(sprintf('tipster_results/figs/centralized_results_k%d.csv',K));
        eps_values = unique(loaded_results(:,2));
        minpts_values = unique(loaded_results(:,3));

        A = zeros(length(eps_values),length(minpts_values) );
%         h = 1;
%         for i=1:length(eps_values)
%             A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 6);
%             h = h + 1;
%         end
        for i=1:length(eps_values)
            %A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 6);
            %loaded_results(bsxfun(@and, loaded_results(:,2)== 25, loaded_results(:,3)== 15), :)
            %h = h + 1;
            for j=1:length(minpts_values)
                valor = loaded_results(bsxfun(@and, loaded_results(:,2)== eps_values(i), loaded_results(:,3)== minpts_values(j)), 6);
                if size(valor, 1) == 0
                   A(i,j)  = 0;
                else
                   A(i,j)  = valor(1,1);
                end
            end            
        end
        
        h = tabularHeatMap(A);
        xlabel('MinPts');
        ylabel('Eps');
        h.YTick =  1:length(eps_values);
        h.YTickLabel =  num2cell(transpose(eps_values));
        h.XTick =  1:length(minpts_values);
        h.XTickLabel = num2cell(transpose(minpts_values));

        %saveas(h, sprintf('tipster_results/centralized_VM_K%d.png',K));
        saveas(h, sprintf('tipster_results/figs/centralized_%s_k%d_results_VM.png',textdatasets{ds},K));
    end
end
%% Distributed
clear;
clc;
%textdatasets = cellstr(['SJMN';'FR  ';'DOE ';'ZF  ';'20ng']);
%textdatasets = cellstr(['SJMN';'FR  ']);
textdatasets = cellstr(['DOE ';'ZF  ';'20ng']);


for ds=1:length(textdatasets)
    for K=[50]
        loaded_results = csvread(sprintf('tipster_results/figs/distributed_%s_k%d_results_ARI_AMI_VM',textdatasets{ds},K));

        eps_values = unique(loaded_results(:,2));
        minpts_values = unique(loaded_results(:,3));

        A = zeros(length(eps_values),length(minpts_values) );
        h = 1;
        for i=1:length(eps_values)
            %A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 6);
            %loaded_results(bsxfun(@and, loaded_results(:,2)== 25, loaded_results(:,3)== 15), :)
            %h = h + 1;
            for j=1:length(minpts_values)
                valor = loaded_results(bsxfun(@and, loaded_results(:,2)== eps_values(i), loaded_results(:,3)== minpts_values(j)), 6);
                if size(valor, 1) == 0
                   A(i,j)  = 0;
                else
                   A(i,j)  = valor(1,1);
                end
            end            
        end
        h = tabularHeatMap(A);
        xlabel('MinPts');
        ylabel('Eps');
        h.YTick =  1:length(eps_values);
        h.YTickLabel =  num2cell(transpose(eps_values));
        h.XTick =  1:length(minpts_values);
        h.XTickLabel = num2cell(transpose(minpts_values));

        %saveas(h, sprintf('tipster_results/figs/distributed_%s_k%d_results_ARI_AMI_VM.png',textdatasets{ds},K));
        saveas(h, sprintf('tipster_results/figs/distributed_%s_k%d_results_VM.png',textdatasets{ds},K));
    end
end
%%
clear;
clc;
K = 50;
loaded_results = csvread(sprintf('results/distribuido_K%d.csv',K));

h = 1;
eps_values = unique(loaded_results(:,2));
minpts_values = unique(loaded_results(:,3));

A = zeros(length(eps_values),length(minpts_values) );

for i=1:length(eps_values)
    A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 6);
    h = h + 1;
end

h = tabularHeatMap(A);

xlabel('MinPts');
ylabel('Eps');
%h.YTick = transpose(eps_values);
h.YTickLabel =  num2cell(transpose(eps_values));
h.XTick =  1:length(minpts_values);
h.XTickLabel = num2cell(transpose(minpts_values));




