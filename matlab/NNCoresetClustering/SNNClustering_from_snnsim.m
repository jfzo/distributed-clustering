function [CORE_PTS_CT, CORE_CLST_CT] =  SNNClustering_from_snnsim(SNN_CT, Eps, MinPts)
    % [CORE_PTS_CT, CORE_CLST_CT] = SNNClustering(SNN, Eps, MinPts)
    %
    % Receives the shared-nearest-neighbors similarity matrix together with
    % Eps and MinPts parameters.
    % Returns two vectors having the same length as rows the SNN matrix
    % has. 
    % CORE_PTS_CT contains in the i-th position:
    %   -1 , if point i is a noise point
    %   0 , if point i is a non-core/non-noise point
    %   1 , if point i is a core-point
    %
    % CORE_CLST_CT contains a 0 for noise points and a value >0 denoting
    % the cluster label of non-core and core-points.
    tic
    
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
    toc

    % Plotting core points with their identified labels
    %scatter(DATA(CORE_PTS_CT,1), DATA(CORE_PTS_CT,2), 5, CORE_CLST_CT(CORE_PTS_CT),'o')
    % Plotting non-core/non-noise points with their identified labels
    %non_core = find(CORE_PTS_CT ~= 0 );
    %if isempty(non_core) > 0
    %    scatter(DATA(non_core,1), DATA(non_core,2), 5, CORE_CLST_CT(non_core),'o')
    %end

    % Export core and non-core/non-noise points with their labels
end