function generate_partition(Nnodes::Int64, N::Int64)
    P = ones(Int64, N)
    rndprm = shuffle(collect(1:N))
    for i = 1:N
        rand_ix = rndprm[i]
        P[rand_ix] = (i % Nnodes == 0)?Nnodes:(i % Nnodes)
        P[rand_ix] += 1 # workers start at 2
    end
    return P
end

function master_work(results::Dict{String,Array{Int32,1}}, inputPath::String, partition::Array{Int64,1}, worker_Eps::Int64, worker_MinPts::Int64, worker_k::Int64, pct_sample::Float64;similarity::String="cosine")
    
    N = length(partition);
    Nnodes = length(unique(partition));
    samples = Dict{Int64, Array{Int64,1}}()    
    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            samples[node_id] = remotecall_fetch( 
            local_work, #function call
            pid,
            worker_assignment,                
            inputPath,
            worker_Eps,
            worker_MinPts,
            worker_k,
            pct_sample,
            similarity=similarity);
        end
    end
    
    # Join sampled core-points
    overall_sample = Array{Int64,1}();
    for i=keys(samples)
        if length(samples[i]) > 0
            overall_sample = vcat(overall_sample, samples[i])
        end        
    end
    sort!(overall_sample);
    
    #println("Overall sample size:",size(overall_sample))
    M, dim = get_header_from_input_file(inputPath);
    D = zeros(dim,length(overall_sample));
    get_slice_from_input_file(D, inputPath, overall_sample)
    
    # Apply SNN-clustering over the centralized points
    cluster_assignment, core_points, cluster_labels, (sil_value, t_Eps, t_MinPts, t_k) = tuned_snn_clustering(D, similarity=similarity)
    
    #=
    cluster_assignment, core_points, cluster_labels = snn_clustering(D,         
        worker_Eps, 
        worker_MinPts,
        min(worker_k, length(overall_sample) - 1),
        similarity = similarity
    )
    =#
    
    
    # Send the labeled core-points, and the assignments to each worker.
    final_result = Dict{Int64, Array{Int64,1}}()    
    # Each worker will label the non-core points by using a KNN-core-point voting scheme.    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            final_result[node_id] = remotecall_fetch( 
            local_work_final, #function call
            pid,
            worker_assignment,
            overall_sample[core_points],
            cluster_assignment[core_points],
            inputPath,
            similarity=similarity);
        end
    end
    
    overall_labels = zeros(Int64, N);
    for i=keys(final_result)
        worker_assignment = find(x -> x==i, partition);
        @assert length(worker_assignment) == length(final_result[i])
        overall_labels[worker_assignment] = final_result[i];
    end
    
    #=
    results["sampledpoints"] = overall_sample;# instance ids relative to the original data array.
    results["assignments"] = cluster_assignment; #cluster labels for each sampled instance.
    results["corepoints"] = core_points;
    results["labels"] = cluster_labels;
    =#
    
    return overall_labels
end


function master_work_dev(results::Dict{String,Array{Int32,1}}, inputPath::String, partition::Array{Int64,1}, worker_Eps::Int64, worker_MinPts::Int64, worker_k::Int64, pct_sample::Float64;similarity::String="cosine")
    Nnodes = length(unique(partition));
    samples = Dict{Int64, Array{Int64,1}}()    
    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            samples[node_id] = remotecall_fetch( 
            local_work_stage1, #function call
            pid,
            worker_assignment,                
            inputPath,
            worker_Eps,
            worker_MinPts,
            worker_k,
            pct_sample,
            similarity=similarity);
        end
    end
    
    # Join sampled core-points
    overall_sample = Array{Int64,1}();
    for i=keys(samples)
        if length(samples[i]) > 0
            overall_sample = vcat(overall_sample, samples[i])
        end        
    end
    sort!(overall_sample)
    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        @async begin
            samples[node_id] = remotecall_fetch( 
            local_work_stage2, #function call
            pid,
            overall_sample,
            worker_assignment,                
            inputPath,
            worker_Eps,
            worker_MinPts,
            worker_k,
            similarity=similarity);
        end
    end
    
    
    #println("Overall sample size:",size(overall_sample))
    M, dim = get_header_from_input_file(inputPath);
    D = zeros(dim,length(overall_sample));
    get_slice_from_input_file(D, inputPath, overall_sample)
    
    # Apply SNN-clustering over the centralized points
    cluster_assignment, core_points, cluster_labels = snn_clustering(D,         
        worker_Eps, 
        worker_MinPts,
        min(worker_k, length(overall_sample) - 1),
        similarity = similarity
    )
    results["sampledpoints"] = overall_sample;# instance ids relative to the original data array.
    results["assignments"] = cluster_assignment; #cluster labels for each sampled instance.
    results["corepoints"] = core_points;
    results["labels"] = cluster_labels;
end
