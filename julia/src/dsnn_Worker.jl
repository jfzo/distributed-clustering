module DSNN_Worker

#using DSNN_IO

function stage1_start(assigned_instances::Array{Int64,1}, 
    inputPath::String, 
    pct_sample::Float64,
    similarity::String="cosine")
    #=
     Takes a file path where the data matrix is storedXX.
    
    It is inmportant that assigned_instances is sorted in ascending order as in this way the indexes will match the 
    column ids of the matrix retrieved by get_slice_from_input_file.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#

    #d = DSNN_IO.load_selected_sparse_matrix(inputPath, assigned_instances)
    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=assigned_instances, l2normalize=true);
    
    k_ap = 200; epsilon = 0.001;
    apix = DSNN_KNN.initialAppGraph(d, k_ap, epsilon, k_ap*2);
    DSNN_KNN.improve_graph!(apix, d, k_ap, epsilon, k_ap*2);
    
    knnmat_ap, nbrhd_len = DSNN_KNN.get_knnmatrix(apix, k, binarize=true)#, sim_threshold = 0.15);
    snnmat_ap = DSNN_KNN.get_snnsimilarity(knnmat_ap, nbrhd_len)
    snn_graph = DSNN_KNN.get_snngraph(knnmat_ap, snnmat_ap);
    results = DSNN_SNN.snn_clustering(0.5, 7, snn_graph);
    

    # storing results
    corepoints = results["corepoints"];
    
    noncorepoints = find(x-> ~(x in results["corepoints"]) && results["labels"][x] > 0, collect(1:length(assigned_instances))); #points not contained in corepoints and also whose label is not Noise.
    
    cluster_assignment = P["cluster_assignment"];#all assignments, noise included
    
    clusters = unique(P["cluster_assignment"][P["assigned_points"]]);#noise not considered
    
    noisy_pts = find(x-> ~(x in P["assigned_points"]),collect(1:length(assigned_instances)));
    
    
    result = Dict{String, Any}();
    result["Eps"] = P["Eps"];
    result["MinPts"] = P["MinPts"];
    result["k"]=P["k"];
    result["clusters"] = clusters
    result["noise_points"] = assigned_instances[noisy_pts]
    result["corepoints"] = assigned_instances[corepoints]
    result["cluster_assignment"] = cluster_assignment #one for each 'assigned_instances' including noise
    
    sample_pts = Int64[]
    for C in clusters
        # find non-core points in the cluster
        noncp_in_cluster = noncorepoints[find(x->cluster_assignment[x]==C, noncorepoints)];
        cp_in_cluster = corepoints[find(x->cluster_assignment[x]==C, corepoints)];
       
        
        # Sample MinPts points from noncp_in_cluster
        sample_C = sample(noncp_in_cluster, ceil(Int64, length(noncp_in_cluster)*pct_sample) , replace=false);
        sample_pts = vcat(sample_pts, sample_C);
    end
    
    #sample_pts = vcat(corepoints, sample_pts);#sampled_pts continas both points
    result["sampled_points"] = assigned_instances[sample_pts];
    result["corepoints"] = assigned_instances[corepoints];
    #assigned_instances[corepoints], 
    return result    
end

function stage2_start(assigned_instances::Array{Int64,1}, 
    overall_corepoints::Array{Int64, 1},
    corepoint_labels::Array{Int64, 1},
    inputPath::String,
    k:Int64,
    KNN::Int64;
    similarity::String="cosine")
    
    
end

end
