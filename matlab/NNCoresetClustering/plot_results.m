% ARI 4
% AMI 5
% VMeasure 6
clear;
clc;
for K=[50, 70, 90]
    loaded_results = csvread(sprintf('results/centralizado_K%d.csv',K));
    eps_values = unique(loaded_results(:,2));
    minpts_values = unique(loaded_results(:,3));

    A = zeros(length(eps_values),length(minpts_values) );
    h = 1;
    for i=1:length(eps_values)
        A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 4);
        h = h + 1;
    end

    h = tabularHeatMap(A);
    xlabel('MinPts');
    ylabel('Eps');
    %h.YTick = transpose(eps_values);
    h.YTickLabel =  num2cell(transpose(eps_values));
    h.XTick =  1:length(minpts_values);
    h.XTickLabel = num2cell(transpose(minpts_values));

    saveas(h, sprintf('results/centralizado_ARI_K%d.png',K));
end

for K=[50, 70, 90]
    loaded_results = csvread(sprintf('results/distribuido_K%d.csv',K));
    eps_values = unique(loaded_results(:,2));
    minpts_values = unique(loaded_results(:,3));

    A = zeros(length(eps_values),length(minpts_values) );
    h = 1;
    for i=1:length(eps_values)
        A(h,:) = loaded_results((loaded_results(:,2)==eps_values(i)), 4);
        h = h + 1;
    end

    h = tabularHeatMap(A);
    xlabel('MinPts');
    ylabel('Eps');
    %h.YTick = transpose(eps_values);
    h.YTickLabel =  num2cell(transpose(eps_values));
    h.XTick =  1:length(minpts_values);
    h.XTickLabel = num2cell(transpose(minpts_values));
    saveas(h, sprintf('results/distribuido_ARI_K%d.png',K));
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




