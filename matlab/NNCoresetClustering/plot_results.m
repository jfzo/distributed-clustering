% ARI 4
% AMI 5
% VMeasure 6

loaded_results = csvread('results/centralizado_K50.csv');;
A = zeros(length(unique(loaded_results(:,2))),length(unique(loaded_results(:,3))) );
h = 1;
for i=1:length(unique(loaded_results(:,2)))
    arr = unique(loaded_results(:,2));
    A(h,:) = loaded_results((loaded_results(:,2)==arr(i)), 4);
    h = h + 1;
end