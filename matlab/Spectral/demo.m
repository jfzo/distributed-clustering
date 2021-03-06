path('../DistributedCoresetAndPCA', path())

%%
clear;
P=importdata('spiral.csv');
Pwclass = P;
P=P(:,1:2);

sigma_1 = 0.1;
sigma_2 = 0.1;
NOISE = mvnrnd([0,0], [sigma_1,0;0,sigma_2], round(size(P,1)));
%P = vertcat(P, NOISE);
P = P + NOISE;
%%
clear;
clc;
Pwclass=importdata('noisy_circles.csv'); % Use parameter $\sigma=0.1$
gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
% adding noise
P = Pwclass(:,1:2);
%%
K = length(unique(Pwclass(:,3))); % it can be manually set also!
A = pdist2(P,P);

N = length(Pwclass(:,3));
RLABELS = unique(Pwclass(:,3));
max_global_score=0;
best_param=-1;
for i=0.75:0.01:3
    try
    %[labels, W] = SpectralClustering(P, 3, i);
    [~, labels, ~] = SpectralClustering(A, K, i, true);
    
    catch ME
        display(sprintf('Errors occurred when computing EVals for parameter sigma=%d',i));
        continue;
    end
    sum_p = 0;
    sum_e = 0;
    
    %for k=unique(labels)
    WLABELS = unique(labels);
     for k=1:length(WLABELS)
        %find the class with max inters.
        maxi_len = 0;
        maxi_class = -1;
        k_labeled = find(labels == WLABELS(k));
        
        for j=1:length(RLABELS)
            j_labeled = find(Pwclass(:,3) == RLABELS(j) );
            int_len = length(intersect(j_labeled, k_labeled) );
            
            if int_len > maxi_len
                maxi_len = int_len;
                maxi_class = RLABELS(j);
            end
        end
        
        sum_p = sum_p + maxi_len;        
        sum_e = sum_e + (length(k_labeled)/N) * log(length(k_labeled)/N);
     end
    
    purity = sum_p / N;
    entropy = -sum_e;
    global_score = (2*purity*(2-entropy))/(purity+(2-entropy));
    if max_global_score < global_score
        max_global_score=global_score;
        best_param=i;
        
    end
end

%[labels, W] = SpectralClustering(P, 3, best_param);

%A = pdist2(P,P);

[centers, labels, W] = SpectralClustering(A, K, best_param, true);
%%
% 
% $$e^{\pi i} + 1 = 0$$
% 

gscatter(P(:,1),P(:,2),labels(:,1))
%%