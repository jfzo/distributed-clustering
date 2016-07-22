function [CORE_PTS_CT, CORE_CLST_CT] = SNNClustering(DATA, K, Eps, MinPts)
    % [CORE_PTS_CT, CORE_CLST_CT] = SNNClustering(DATA, Knn, Eps, MinPts)
    %
    % Returns two vectors having the same length as rows the DATA matrix
    % has. 
    % CORE_PTS_CT contains in the i-th position:
    %   -1 , if point i is a noise point
    %   0 , if point i is a non-core/non-noise point
    %   1 , if point i is a core-point
    %
    % CORE_CLST_CT contains a 0 for noise points and a value >0 denoting
    % the cluster label of non-core and core-points.

    [KNN_CT, SNN_CT] = compute_knn_snn(DATA, K);

    display('Sparsification complete. Starting SNN density computation...')
    
    % Counting close points (in terms of SNN similarity) for each point ~ Density
    DST_CT = zeros(length(SNN_CT) + 1, 1); %array to store density
    for i=1:length(SNN_CT)
        dense_ng = find(SNN_CT{i} > Eps);
        DST_CT(i) = DST_CT(i) + length(dense_ng);
        for dense_item=1:length(dense_ng) %updates the counters of the neighb.
            DST_CT(i+dense_ng(dense_item)) = DST_CT(i+dense_ng(dense_item)) + 1;
        end
    end

    % Identifying CORE points

    display('Density computation complete. Starting CORE point identification...')
    
    % snn similarity function: e.g. snn_sim(SNN{1},7,1)
    snn_sim = @(SNN_info, i, j) SNN_info{min(i,j)}(abs(i-j));
    CORE_PTS_CT = DST_CT > MinPts;
    CORE_CLST_CT = zeros(length(DST_CT),1);
    core_points = find(DST_CT > MinPts);
    clst_id = 0;
    for i=1:size(core_points, 1)-1 %identifying clusters
        if CORE_CLST_CT(core_points(i)) == 0
            clst_id = clst_id + 1;
            CORE_CLST_CT(core_points(i)) = clst_id;
        end
        for j=i+1:size(core_points, 1)
            if snn_sim(SNN_CT,core_points(i),core_points(j)) >= Eps
                CORE_CLST_CT(core_points(j)) = CORE_CLST_CT(core_points(i));
            end
        end
    end

    display('Core points identified. Starting noise removal (final step).')
    % Discarding Noise points (marked with -1 in its entry in CORE_PTS_CT)
    non_core = find(CORE_PTS_CT == 0 );
    core = find(CORE_PTS_CT == 1);
    core_clst_id = CORE_CLST_CT; % copy made to enable parallel loop
    for i=1:length(non_core)
        % find nearest core point
        nearest_sim = 0;
        nearest_core = -1;
        for j=1:length(core)
            j_sim = snn_sim(SNN_CT, non_core(i), core(j));
            if j_sim > nearest_sim
                nearest_sim = j_sim;
                nearest_core = core(j);
            end
        end
        % if the snn similarity is lower than Eps => discard the point
        if nearest_sim < Eps
            CORE_PTS_CT(non_core(i)) = -1;
        else % otherwise, label the point with the nearest core point label.
            CORE_CLST_CT(non_core(i)) = core_clst_id(nearest_core);
        end
    end


    % Plotting core points with their identified labels
    %scatter(DATA(CORE_PTS_CT,1), DATA(CORE_PTS_CT,2), 5, CORE_CLST_CT(CORE_PTS_CT),'o')
    % Plotting non-core/non-noise points with their identified labels
    %non_core = find(CORE_PTS_CT ~= 0 );
    %if isempty(non_core) > 0
    %    scatter(DATA(non_core,1), DATA(non_core,2), 5, CORE_CLST_CT(non_core),'o')
    %end

    % Export core and non-core/non-noise points with their labels
end

% % parameters
% % Eps y knn
% Eps = 20;
% knn = 20;
% %%%%
% clear;
% clc;
% Pwclass=importdata('noisy_circles.csv'); % Use parameter $\sigma=0.1$
% gscatter(Pwclass(:,1),Pwclass(:,2),Pwclass(:,3))
% % adding noise
% P = Pwclass(:,1:2);
% %% Building and sparsifying the distance matrix
% A = pdist2(P,P);
% 
% for row=1:size(A,1)
%     [Sval, Sind] = sort(A(row,:),'ascend');
%     A(row, Sind(knn+2:end)) = 0; % consider self-similarity
% end
% %% SNN graph
% B = zeros(size(A));
% for row_i=1:(size(A,1)-1)
%     for row_j=(row_i+1):size(A,1)
%         B(row_i, row_j) = sum((A(row_i,:) > 0) & (A(row_j,:) > 0));
%     end
% end
% display('Ok!');
% 
% %% CURE DATASET GENERATOR
% clc;
% % The big circle
% mu = [0 -0.5];
% Sigma = [5 0; 0 1]; R = chol(Sigma);
% n1 = 3000;
% z1 = repmat(mu,n1,1) + randn(n1,2)*R;
% l1 = zeros(n1,1) + 1;
% 
% %the 1st elipse and its bridge
% mu = [-2 6];
% Sigma = [3 0; 0 0.05]; R = chol(Sigma);
% n2 = 1000;
% z2 = repmat(mu,n2,1) + randn(n2,2)*R;
% l2 = zeros(n2,1) + 2;
% 
% 
% mu = [1.0 5.9];
% Sigma = [20 0; 0 0.01]; R = chol(Sigma);
% n6 = 100;
% z6 = repmat(mu,n6,1) + rand(n6,2)*R;
% l6 = zeros(n6,1) + 2;
% 
% %the 2nd elipse
% mu = [10 6];
% Sigma = [3 0; 0 0.05]; R = chol(Sigma);
% n3 = 1000;
% z3 = repmat(mu,n3,1) + randn(n3,2)*R;
% l3 = zeros(n3,1) + 3;
% 
% %the upper small circle
% mu = [15 1];
% Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
% n4 = 1000;
% z4 = repmat(mu,n4,1) + randn(n4,2)*R;
% l4 = zeros(n4,1) + 4;
% 
% %the lower small circle
% mu = [15 -2];
% Sigma = [0.3 0; 0 0.05]; R = chol(Sigma);
% n5 = 1000;
% z5 = repmat(mu,n5,1) + randn(n5,2)*R;
% l5 = zeros(n5,1) + 5;
% 
% % background noise
% mu = [-8 -5];
% Sigma = [700 0; 0 150]; R = chol(Sigma);
% n7 = 1000;
% z7 = repmat(mu,n7,1) + rand(n7,2)*R;
% l7 = zeros(n7,1) + 7;
% 
% % merging 
% Z = vertcat(z1, z2, z3, z4, z5,z6, z7);
% labels = vertcat(l1,l2,l3,l4,l5,l6,l7);
% gscatter(Z(:,1),Z(:,2), labels )
