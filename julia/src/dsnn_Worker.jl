module DSNN_Worker

using DSNN_IO
using DSNN_KNN
using DSNN_SNN
using Stats

function stage1_start(
        assigned_instances::Array{Int64,1}, 
        inputPath::String, 
        config_params::Dict{String, Any}
    )
    #=
     Takes a file path where the data matrix is storedXX.
    
    It is inmportant that assigned_instances is sorted in ascending order as in this way the indexes will match the 
    column ids of the matrix retrieved by get_slice_from_input_file.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#


    pct_sample=config_params["worker.sample_pct"];
    knn=config_params["worker.knn"];

    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=assigned_instances, l2normalize=true);
    
    snnmat, knnmat = DSNN_KNN.get_snnsimilarity(d, knn, relative_values=false, l2knng_path=config_params["l2knng.path"]);

    adj_mat = snnmat;
    if config_params["worker.use_snngraph"]
        println("[W] Using the SNN Graph as Adjacency Matrix");
        snn_graph = DSNN_KNN.get_snngraph(knnmat, snnmat);
        #snn_graph = DSNN_KNN.get_snngraph(knnmat);# raw cosine similarity as edge weight
        adj_mat = snn_graph;
    end
    result = Dict{String, Any}();
    result["knn"]=knn;

    
    if config_params["worker.use_snnclustering"]
        snn_eps=config_params["worker.snn_eps"];
        snn_minpts=config_params["worker.snn_minpts"];    
        result["Eps"] = snn_eps;
        result["MinPts"] = snn_minpts;

        
        println("[W] executing snn clustering with eps:",snn_eps," and minpts:", snn_minpts)
        #cl_results = DSNN_SNN.snn_clustering(snn_eps, snn_minpts, snn_graph);
        cl_results = DSNN_SNN.snn_clustering(snn_eps, snn_minpts, adj_mat);

        cl_labels = cl_results["labels"];# Matrix containing length(assigned_instances) x num_clusters_found
        cl_clusters = cl_results["clusters"];# Array with the label of each column of the matrix above
        cl_corepoints = cl_results["corepoints"];# Array with data point indexes

        if length(cl_results["corepoints"]) == 0
            println("[W] Warning! No corepoints were found. Aborting execution in this worker.");
            error(@sprintf("No corepoints were found by this worker (%d)", myid()) )
        else
            println(@sprintf("[W] Nr. corepoints found by this worker (%d):%d", myid(),length()cl_results["corepoints"]) )
            
        end

        noise_col = find(x->x==DSNN_SNN.NOISE, cl_clusters);#cl_labels[:,noise_col].nzind contains all the noisy point indexes
        noisy_pts = Int64[];
        if length(noise_col) > 0 
            noisy_pts = cl_labels[:,noise_col[1]].nzind;
        end
        noncorepoints = find(x->~(x in cl_corepoints) && ~(x in noisy_pts), collect(1:length(assigned_instances))); #cl_labels[:,noise_col].nzind contains all the noisy point indexes
        #
        # All these arrays contain point id's relative to assigned_instances!
        #

        # Building a subset of assigned_points with the corepoints and a small sample of points with each one

        result["clusters"] = cl_clusters;
        result["corepoints"] = assigned_instances[cl_corepoints];
        result["noncorepoints"] = assigned_instances[noncorepoints];
        result["noise_points"] = assigned_instances[noisy_pts];
        result["cluster_assignment"] = cl_labels;

        RNG_W = srand(config_params["seed"]);
        sample_pts = Int64[]
        for C in eachindex(cl_clusters)
            # Sample points from cl_labels[:,C].nzind
            # wv = weights([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            C_size = length(cl_labels[:,C].nzind);
            sample_C = Stats.sample(RNG_W, cl_labels[:,C].nzind, ceil(Int64, C_size*pct_sample), replace=false);        
            sample_pts = vcat(sample_pts, sample_C);
        end

        # filter corepoints from the sample sample_pts
        filtered_sample = Int64[];
        for pt in sample_pts
            if !(pt in cl_corepoints)
                push!(filtered_sample, pt);
            end
        end

        sort!(filtered_sample);
        #sample_pts = vcat(corepoints, sample_pts);#sampled_pts contains both points
        result["sampled_points"] = assigned_instances[filtered_sample];
    else
        snn_eps=config_params["worker.coredetection_eps"];
        snn_minpts=config_params["worker.coredetection_minpts"];
        result["Eps"] = snn_eps;
        result["MinPts"] = snn_minpts;
        println("[W] executing corepoint detection with eps:",snn_eps," and minpts:", snn_minpts)        
        corepts, sample_pts = DSNN_SNN.get_dataclusters_sample(snn_eps, snn_minpts, adj_mat, pct_sample);
        
        if length(corepts) == 0
            println("[W] Warning! No corepoints were found. Aborting execution in this worker.");
            error(@sprintf("No corepoints were found by this worker (%d)", myid()) )
        end
        
        result["sampled_points"] = assigned_instances[sample_pts];
        result["corepoints"] = assigned_instances[corepts];
        result["noise_points"] = Int64[];
        result["cluster_assignment"] = Int64[];
    end
    
    # Operation condition: no more than 30% of the total data assigned is reported
    if length(result["corepoints"]) > (0.5*length(assigned_instances))
        println("[W] Warning! Too many corepoints were found. Aborting execution in this worker.");
        error(@sprintf("Too many corepoints %d from %d (more than 30 pct) were found by this worker (%d)", length(result["corepoints"]),length(assigned_instances),myid()) )
    end

    return result
end




"""
"""
function stage2_start(assigned_instances::Array{Int64,1}, 
    overall_corepoints::Array{Int64, 1},
    corepoint_labels::Array{Int64, 1},
    inputPath::String,
    config_params::Dict{String, Any})
 
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
    # useful for snn
    #iscore = fill(false, length(instances));
    
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

    #println("[W] Labeling assigned instances from the oveall corepoints (value for k:",round(Int64,0.5*k),")")
    #snnmat, knnmat = DSNN_KNN.get_snnsimilarity(d, round(Int64,0.5*k), min_threshold=0.7, l2knng_path=config_params["l2knng.path"]);

    knn = config_params["worker.knn"];

    println("[W] Labeling assigned instances from the overall corepoints")
    snnmat, knnmat = DSNN_KNN.get_snnsimilarity(d, knn, l2knng_path=config_params["l2knng.path"]);

    adj_mat = snnmat;
    if config_params["worker.use_snngraph"]
        println("[W] Using the SNN Graph as Adjacency Matrix");
        snn_graph = DSNN_KNN.get_snngraph(knnmat, snnmat);
        #snn_graph = DSNN_KNN.get_snngraph(knnmat);# raw cosine similarity as edge weight
        adj_mat = snn_graph;
    end

  
    # Label assignment in the style of snn
    # This is the last step in the snn-clustering algorithm
    for i in collect(1:length(instances))
        if is_corepoint(instances[i])
            labels[instances[i]] = corepoint_labels[instances[i]];
            continue
        end
        # get the nearest corepoint
        nst_core = -1;
        nst_core_sim = -1;
        labels[instances[i]] = DSNN_SNN.NOISE;        
        for q in adj_mat[:,i].nzind
            if is_corepoint(instances[q])  # consider only corepoint neighbors
                if adj_mat[q,i] > nst_core_sim
                    nst_core = q;
                    nst_core_sim = adj_mat[q,i];
                end
            end
        end
        if nst_core_sim >= config_params["worker.snn_eps"]
            labels[instances[i]] = corepoint_labels[instances[nst_core]];
        end
    end
    
    # Filter the labels of the non core points only.
    assignment_labels = fill(-1, length(assigned_instances));
    for i in eachindex(assigned_instances)
        #if haskey(labels, assigned_instances[i])
        assignment_labels[i] = labels[assigned_instances[i]];
        #end
    end
    
    return Dict{String, Array{Int64,1}}("assigned_instances" => assigned_instances, "labels" => assignment_labels);
end

end
