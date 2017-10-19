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

    d = DSNN_IO.load_selected_sparse_matrix(inputPath, assigned_instances)
    
    #cluster_assignment, core_points, cluster_labels = snn_clustering(d, Eps, MinPts, k, similarity=similarity)
    
    P =  tuned_snn_clustering(d,  
    worker_eps_start_val,
    worker_eps_step_val,
    worker_eps_end_val,
    worker_minpts_start_val,
    worker_minpts_step_val,
    worker_minpts_end_val,
    k_range, similarity);
    

    # storing results
    corepoints = P["corepoints"];
    
    noncorepoints = find(x-> ~(x in P["corepoints"]) && P["cluster_assignment"][x] > 0, collect(1:length(assigned_instances))); #points not contained in corepoints and also whose label is not Noise.
    
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

end
