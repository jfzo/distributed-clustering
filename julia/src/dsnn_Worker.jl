module DSNN_Worker

using DSNN_IO
using DSNN_KNN
using DSNN_SNN
using Stats

function stage1_start(assigned_instances::Array{Int64,1}, 
    inputPath::String, 
    pct_sample::Float64,
    k::Int64;
    k_ap::Int64 = 200,
    snn_eps::Float64 = 0.4,
    snn_minpts::Int64 = 10
    )
    #=
     Takes a file path where the data matrix is storedXX.
    
    It is inmportant that assigned_instances is sorted in ascending order as in this way the indexes will match the 
    column ids of the matrix retrieved by get_slice_from_input_file.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#

    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=assigned_instances, l2normalize=true);
    
    #k_ap = 200; 
    epsilon = 0.01;
    apix = DSNN_KNN.initialAppGraph(d, k_ap, epsilon, k_ap*2);
    DSNN_KNN.improve_graph!(apix, d, k_ap, epsilon, k_ap*2);
    
    knnmat_ap, nbrhd_len = DSNN_KNN.get_knnmatrix(apix, k, binarize=true)#, sim_threshold = 0.15);
    snnmat_ap = DSNN_KNN.get_snnsimilarity(knnmat_ap, nbrhd_len)
    snn_graph = DSNN_KNN.get_snngraph(knnmat_ap, snnmat_ap);

    println("[W] executing snn clustering with eps:",snn_eps," and minpts:", snn_minpts)
    cl_results = DSNN_SNN.snn_clustering(snn_eps, snn_minpts, snn_graph);

    cl_labels = cl_results["labels"];# Matrix containing length(assigned_instances) x num_clusters_found
    cl_clusters = cl_results["clusters"];# Array with the label of each column of the matrix above
    cl_corepoints = cl_results["corepoints"];# Array with data point indexes
    
    noise_col = find(x->x==DSNN_SNN.NOISE, cl_clusters)[1]; #cl_labels[:,noise_col].nzind contains all the noisy point indexes
    noisy_pts = cl_labels[:,noise_col].nzind;
    noncorepoints = find(x->~(x in cl_corepoints) && ~(x in noisy_pts), collect(1:length(assigned_instances))); #cl_labels[:,noise_col].nzind contains all the noisy point indexes
    #
    # All these arrays contain point id's relative to assigned_instances!
    #
    
    # Building a subset of assigned_points with the corepoints and a small sample of points with each one
    
    result = Dict{String, Any}();
    result["Eps"] = snn_eps;
    result["MinPts"] = snn_minpts;
    result["k"]=k;
    result["clusters"] = cl_clusters;
    result["corepoints"] = assigned_instances[cl_corepoints];
    result["noncorepoints"] = assigned_instances[noncorepoints];
    result["noise_points"] = assigned_instances[noisy_pts];
    result["cluster_assignment"] = cl_labels;
    
    sample_pts = Int64[]
    for C in eachindex(cl_clusters)
        # Sample points from cl_labels[:,C].nzind
        # wv = weights([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        C_size = length(cl_labels[:,C].nzind);
        sample_C = Stats.sample(cl_labels[:,C].nzind, ceil(Int64, C_size*pct_sample), replace=false);
        sample_pts = vcat(sample_pts, sample_C);
    end
    
    sort!(sample_pts);
    #sample_pts = vcat(corepoints, sample_pts);#sampled_pts contains both points
    result["sampled_points"] = assigned_instances[sample_pts];
    return result    
end

function stage2_start(assigned_instances::Array{Int64,1}, 
    overall_corepoints::Array{Int64, 1},
    corepoint_labels::Array{Int64, 1},
    inputPath::String,
    k::Int64)
 
    s_instances = Set(vcat(assigned_instances, overall_corepoints)); # used to fusion all the points that are going to be used in this function
    instances = sort(collect(s_instances));# an ordered list of data point ids to load from disk (contains previously assigned points and the list of overall corepoints
    
    #println("Overall corepoints and their labels: ",length(overall_corepoints)," == ",length(corepoint_labels));
    #assert(length(overall_corepoints) == length(corepoint_labels));

    # Dict with a corepoint id and its label (id's given originally to this function)
    corepoint_labels = Dict{Int64, Int64}(zip(overall_corepoints, corepoint_labels));
    is_corepoint = function(p) return haskey(corepoint_labels, p) end; # more confortable way of verifying if a point p is contained into the corepoints list.


    #labels = fill(-1, length(s_instances));
    labels = Dict{Int64, Int64}();
    cp_in_data = Int64[]; # array contaning all corepoints relative to this fusioned dataset matrix columns
    ncp_in_data = Int64[];
    for i in eachindex(instances)
        # default label set to -1 when the data point is not a corepoint
        if is_corepoint(instances[i])
            #labels[i] = corepoint_labels[instances[i]];
            push!(cp_in_data, i);
        else
            push!(ncp_in_data, i);
            labels[instances[i]] = -1;
        end
    end

    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=instances, l2normalize=true);

    k_ap = 200; epsilon = 0.01;
    apix = DSNN_KNN.initialAppGraph(d, k_ap, epsilon, k_ap*2);
    DSNN_KNN.improve_graph!(apix, d, k_ap, epsilon, k_ap*2);
    
    knnmat_ap, nbrhd_len = DSNN_KNN.get_knnmatrix(apix, k, binarize=true)#, sim_threshold = 0.15);
    snnmat_ap = DSNN_KNN.get_snnsimilarity(knnmat_ap, nbrhd_len)
    snn_graph = DSNN_KNN.get_snngraph(knnmat_ap, snnmat_ap); # adjacency matrix in which two vertices are connected only if both are in each other neighborhood
    #snn_graph[:, cp_in_data] denote all the columns containing sim values between corepoints and the remaining points.

    nrst_cp = Dict{Int64, Tuple{Int64, Float64}}();
    # Strategy followed to label each noncore data point:
    # Traverse each corepoint column and assign its label to the nzind that are not corepoints
    # WARN: Maybe some points have 0-similarity value against every corepoint
    for i in cp_in_data
        nnz_nbrs = snn_graph[:,i].nzind
        for j in nnz_nbrs
            if !is_corepoint(instances[j])
                if !haskey(nrst_cp, j)
                    nrst_cp[j] = (i, snn_graph[j,i]);
                else
                    if nrst_cp[j][2] < snn_graph[j,i]
                        nrst_cp[j] = (i,snn_graph[j,i])
                    end
                end
            end
        end
    end

    # assign the nearest corepoint label into the label array
    #println("Noncorepoints examined: ",length(keys(nrst_cp))," / should be equal to ",length(ncp_in_data))
    for i in eachindex(instances)
        if !is_corepoint(instances[i])
            if haskey(nrst_cp, i)
                nrst_cp_ix = instances[nrst_cp[i][1]];
                labels[instances[i]] = corepoint_labels[nrst_cp_ix];
            end # otherwise the label is set to a negative number
        end
    end

    # Filter the labels of the non core points only.
    assignment_labels = fill(-1, length(assigned_instances));
    for i in eachindex(assigned_instances)
        if haskey(labels, assigned_instances[i])
            assignment_labels[i] = labels[assigned_instances[i]];
        end
    end
    
    return Dict{String, Array{Int64,1}}("assigned_instances" => assigned_instances, "labels" => assignment_labels);
    #return Dict{Int64, Int64}(zip(assigned_instances, assignment_labels));
end

end
