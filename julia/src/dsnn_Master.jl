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
#using LightGraphs
using Graphs

function start(results::Dict{String, Any}, 
    inputPath::String, 
    partition::Array{Int64,1}, 
    config_params::Dict{String, Any}
    )
    
    N = length(partition);
    Nnodes = length(unique(partition));
    samples = Dict{Int64, Array{Int64,1}}()
    worker_result = Dict{Int64, Dict{String, Any}}()

    knn = config_params["worker.knn"];
    
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
            config_params
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
    
    #s_sampled_data = Set(sampled_data); # used to fusion all the points that are going to be used in this function
    #sampled_data = sort(collect(s_sampled_data));# an ordered list of data point ids to load from disk (contains previously assigned     
    sort!(sampled_data);
    
    results["stage1_corepoints"] = overall_sample_corepoints; #Only corepoint data retrieved from the workers
    results["stage1_sampled"] = overall_sample; #Only noncorepoint data retrieved from the workers
    #results["centralized_worker_points"] = sampled_data; #All the retrieved data from the workers
   
    sampled_data = results["stage1_corepoints"];

    println("[M] Corepoints (",length(overall_sample_corepoints),") and Samples (",length(overall_sample),")") 



    ###
    ### STAGE 2
    ###
    
    stage2_knn = config_params["master.stage2knn"];
    
    #recall that sampled-data contains corepoints and their samples
    d = DSNN_IO.sparseMatFromFile(inputPath, assigned_instances=sampled_data, l2normalize=true);
    num_points = size(d, 2);
    
    snnmat, knnmat = DSNN_KNN.get_snnsimilarity(d, stage2_knn, l2knng_path=config_params["l2knng.path"]);
    snn_graph = DSNN_KNN.get_snngraph(knnmat, snnmat);
    
    assert(num_points == size(snn_graph,2));
    
    #=    
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
    =#

    adj_mat = snn_graph;
    numpoints = size(snn_graph, 1);
    G = Graphs.simple_adjlist(numpoints, is_directed=false);
    for i in collect(1:numpoints)
        for j in adj_mat[:,i].nzind
            Graphs.add_edge!(G, i, j)
        end
    end

    cmps = Graphs.connected_components(G);

    println("Num. connected components:",length(cmps));
    labels_found = fill(-1, numpoints);
    for cmp_i in eachindex(cmps)
        for p in cmps[cmp_i]
            labels_found[p] = cmp_i;
        end
    end
    corepoint_labels = labels_found[find(x->x in overall_sample_corepoints, sampled_data)]; 

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
            config_params);
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
