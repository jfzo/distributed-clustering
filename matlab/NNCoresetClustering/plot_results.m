% ARI 4
% AMI 5
% VMeasure 6

for K=[50, 70, 90]
    loaded_results = csvread(sprintf('results/centralizado_K%d.csv',K));
    A = zeros(length(unique(loaded_results(:,2))),length(unique(loaded_results(:,3))) );
    h = 1;
    for i=1:length(unique(loaded_results(:,2)))
        arr = unique(loaded_results(:,2));
        A(h,:) = loaded_results((loaded_results(:,2)==arr(i)), 6);
        h = h + 1;
    end

    h = tabularHeatMap(A);
    h.XTickLabel =  unique(loaded_results(:,2));
    h.YTickLabel = unique(loaded_results(:,3));
    saveas(h, sprintf('results/centralizado_VM_K%d.png',K));
end

for K=[50, 70, 90]
    loaded_results = csvread(sprintf('results/distribuido_K%d.csv',K));
    A = zeros(length(unique(loaded_results(:,2))),length(unique(loaded_results(:,3))) );
    h = 1;
    for i=1:length(unique(loaded_results(:,2)))
        arr = unique(loaded_results(:,2));
        A(h,:) = loaded_results((loaded_results(:,2)==arr(i)), 6);
        h = h + 1;
    end

    h = tabularHeatMap(A);
    h.XTickLabel =  unique(loaded_results(:,2));
    h.YTickLabel = unique(loaded_results(:,3));
    saveas(h, sprintf('results/distribuido_VM_K%d.png',K));
end