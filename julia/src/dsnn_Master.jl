module DSNN_Master
function generate_partitions(Nnodes::Int64, N::Int64)
    P = ones(Int64, N)
    rndprm = shuffle(collect(1:N))
    for i = 1:N
        rand_ix = rndprm[i]
        P[rand_ix] = (i % Nnodes == 0)?Nnodes:(i % Nnodes)
        P[rand_ix] += 1 # workers start at 2
    end
    return P
end

include("/workspace/distributed_clustering/julia/src/dsnn_Worker.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_KNN.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
using LightGraphs

function start(results::Dict{String, Any}, 
    inputPath::String, 
    partition::Array{Int64,1}, 
    pct_sample::Float64,
    snn_cut_point::Int64;
    worker_params::Dict{String, Any}=Dict{String,Any}("k"=>100, "snn_eps"=>0.5, "snn_minpts"=>5, "k_appindex"=>200)
    )
    
    N = length(partition);
    Nnodes = length(unique(partition));
    samples = Dict{Int64, Array{Int64,1}}()
    worker_result = Dict{Int64, Dict{String, Any}}()
    
    ###
    ### STAGE 1
    ###
    println("[M] Starting Stage 1 (assignment distribution and corepoint identification)");
    @sync for (idx, pid) in enumerate(workers())
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            worker_result[node_id] = remotecall_fetch( 
            DSNN_Worker.stage1_start, #function call
            pid,
            worker_assignment,                
            inputPath,
            pct_sample,
            worker_params["k"],
            k_ap = worker_params["k_appindex"],
            snn_eps = worker_params["snn_eps"],
            snn_minpts = worker_params["snn_minpts"]
        );
        end
    end
    

    println("[M] Joining worker's results of Stage 1");
    # Join sampled points
    overall_sample = Array{Int64,1}(); # contains real instance ids
    overall_sample_corepoints = Array{Int64,1}(); # contains real instance ids
    sampled_data = Array{Int64,1}(); # contains real instance ids
    for i=keys(worker_result)
        if length(worker_result[i]["sampled_points"]) > 0
            
            overall_sample = vcat(overall_sample, worker_result[i]["sampled_points"])
            
            overall_sample_corepoints = vcat(overall_sample_corepoints, worker_result[i]["corepoints"])
            
            sampled_data = vcat(sampled_data, worker_result[i]["sampled_points"]);
            sampled_data = vcat(sampled_data, worker_result[i]["corepoints"]);
            println("Amount of noisy data points detected by worker ",i,":",length(worker_result[i]["noise_points"]));
        end        
    end
    sort!(overall_sample);    
    sort!(overall_sample_corepoints);
    sort!(sampled_data);
    
    results["stage1_corepoints"] = overall_sample_corepoints; #Only corepoint data retrieved from the workers
    results["stage1_sampled"] = overall_sample; #Only noncorepoint data retrieved from the workers
    #results["centralized_worker_points"] = sampled_data; #All the retrieved data from the workers
    
    #=
    # Code block that gets worker-id for each gathered corepoint.
    overall_sample_workers = fill(-1, length(overall_sample_corepoints))
    for (idx, pid) in enumerate(workers())
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        for i=collect(1:length(overall_sample_corepoints))
            if overall_sample_corepoints[i] in worker_assignment
                overall_sample_workers[i] = node_id - 1
            end
        end
    end
    results["centralized_worker_corepoints_origin"] = overall_sample_workers
    =#
    
    println("[M] Corepoints (",length(overall_sample_corepoints),") and Samples (",length(overall_sample),")") 



    ###
    ### STAGE 2
    ###
    
    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=sampled_data, l2normalize=true);
    num_points = size(d, 2);
    
    #k_ap = worker_params["k_appindex"]; epsilon = 0.001;#epsilon is set always to this value
    #apix = DSNN_KNN.initialAppGraph(d, k_ap, epsilon, k_ap*2);
    #DSNN_KNN.improve_graph!(apix, d, k_ap, epsilon, k_ap*2);
    
    k = worker_params["k"];
    #knnmat_ap, nbrhd_len = DSNN_KNN.get_knnmatrix(apix, k, binarize=true); #, sim_threshold = 0.15);
    #snnmat_ap = DSNN_KNN.get_snnsimilarity(knnmat_ap, nbrhd_len)
    #snn_graph = DSNN_KNN.get_snngraph(knnmat_ap, snnmat_ap);

    snnmat, knnmat = DSNN_KNN.get_snnsimilarity(d, k, l2knng_path="/workspace/l2knng/build/knng");
    snn_graph = DSNN_KNN.get_snngraph(knnmat, snnmat);
    
    assert(num_points == size(snn_graph,2));
    
    println("[M] Creating SNN graph with data retrieved from workers...")
    #println("[M] Creating SNN graph with data retrieved from workers...(cut_point set to ",snn_cut_point,")")
    G = LightGraphs.Graph(num_points)
    for i in collect(1:num_points)
       for j in snn_graph[:, i].nzind
           if j > i 
               # maybe a threshold based on snn_graph[j,i] could be used !
               if ~LightGraphs.add_edge!(G, i, j)
                   println("[M] Error: Cannot add edge between vertices ",i," and ",j)
               end
           end
       end
    end
    vLN, conv_history = LightGraphs.label_propagation(G);
    # getting the corepoint labels
    # find positions in sampled_data that correpond to the gathered corepoints
    println("[M] Number of groups detected with retrieved data:",length(unique(vLN)));
    corepoint_labels = vLN[find(x->x in overall_sample_corepoints, sampled_data)];
    
    #assert(length(corepoint_labels) == length(overall_sample_corepoints))
    
    results["stage1_graph"] = snn_graph;
    
    results["centralized_worker_corepoints_labels"] = corepoint_labels;
    
    println("[M] Retransmitting overall corepoints (Stage 2)...") 
    
    # Send the labeled core-points, and the assignments to each worker.
    final_result = Dict{Int64, Dict{String, Array{Int64,1}}}()    
    # Each worker will label the non-core points by using a KNN-core-point voting scheme.    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            final_result[node_id] = remotecall_fetch( 
            DSNN_Worker.stage2_start, #function call
            pid,
            worker_assignment,
            overall_sample_corepoints,
            corepoint_labels,
            inputPath,
            worker_result[node_id]["k"]);
        end
    end
    
    
    
    println("[M] Joining Worker's results of Stage 2.") 

    
    overall_labels = zeros(Int64, N);
    for worker_id=keys(final_result)
        # final_result[worker_id] is a Dict with the assigned instances to that worker and the corresponding labels!
        #worker_assignment = find(x -> x==worker_id, partition);
        #@assert length(worker_assignment) == length(final_result[i])
        worker_assignment = final_result[worker_id]["assigned_instances"];
        worker_assignment_labels = final_result[worker_id]["labels"];
        overall_labels[worker_assignment] = worker_assignment_labels;
    end
        
    println("[M] Generating worker's final labelings...") 
    #results["corepoints"] = overall_sample[corepoints_ix];
    results["stage2_labels"] = overall_labels;
end
end
