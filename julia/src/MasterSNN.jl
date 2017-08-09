#module MasterSNN

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

function exec_cluto_rb(vectors_file::String, nclusters::Int64; CLUTOV_CMD::String="./cluto-2.1.2/Linux-x86_64/vcluster"
)
    #=
    Executes the Repeated Bisection algorithm (similar to kmeans) from the Cluto framework.
    Requires the number of clusters to find as a parameter
    =#
    output = readstring(`$CLUTOV_CMD -clustfile=$vectors_file.k$nclusters $vectors_file $nclusters`);
    
    assign_fpath=@sprintf("%s.k%d", vectors_file,nclusters)
    
    f = open(assign_fpath);
    labels=Int64[];
    for ln in eachline(f)
        lbl_i=parse(Int64, ln);
        push!(labels, lbl_i);
    end
    close(f)
    return labels
end

#function master_work(results::Dict{String, Any}, inputPath::String, partition::Array{Int64,1}, pct_sample::Float64;similarity::String="cosine", K::Int64=50, KNN::Int64=3,Eps_range = collect(5.0:5.0:50.0), MinPts_range = collect(20:10:50),k_range = [40, 50], snn_cut_point::Int64=5)
function master_work(results::Dict{String, Any}, inputPath::String, partition::Array{Int64,1}, pct_sample::Float64;similarity::String="cosine", K::Int64=50, KNN::Int64=3,snn_cut_point::Int64=5)
    
    N = length(partition);
    Nnodes = length(unique(partition));
    samples = Dict{Int64, Array{Int64,1}}()
    worker_result = Dict{Int64, Dict{String, Any}}()
    
    @sync for (idx, pid) in enumerate(workers())
        #println(idx,' ', pid)
        node_id = idx+1;
        worker_assignment = find(x -> x==node_id, partition);
        sort!(worker_assignment)
        @async begin
            worker_result[node_id] = remotecall_fetch( 
            local_work, #function call
            pid,
            worker_assignment,                
            inputPath,
            pct_sample,
            similarity=similarity);
        end
    end
    

    
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
            
        end        
    end
    sort!(overall_sample);    
    sort!(overall_sample_corepoints);
    sort!(sampled_data);
    
    results["centralized_worker_corepoints"] = overall_sample_corepoints; #Only corepoint data retrieved from the workers
    results["centralized_worker_samples"] = overall_sample; #Only noncorepoint data retrieved from the workers
    results["centralized_worker_points"] = sampled_data; #All the retrieved data from the workers
    
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
    
    println("Joining worker's corepoints (",length(overall_sample_corepoints),") and worker's samples (",length(overall_sample),")") 

    ### Comment unless premature output is needed (DEBUG PURPOSES)
    #return overall_sample
    
    ### UNDER TESTING
    #### THIS PART IS NOT WORKING...MAYBE FINDING CLUSTERS FROM SAMPLED OR COREPOINTS BASED ON SNN IS NOT A GOOD IDEA!
    #println("Overall sample size:",size(overall_sample))
    M, dim = get_header_from_input_file(inputPath);
    D = zeros(dim,length(sampled_data));
    get_slice_from_input_file(D, inputPath, sampled_data)
    
    
    #### ALTERNATIVE
    num_points = size(D,2);
    cent_data = D;

    Snn_cent = zeros(Float64, num_points, num_points);
    S_cent = zeros(Float64, num_points, num_points);
    compute_similarities(cent_data, K, Snn_cent, S_cent, similarity=similarity);

    println("Creating SNN graph with data retrieved from workers...(cut_point set to ",snn_cut_point,")")
    G = Graph(num_points)
    #G_weights = Float64[];
    for i=collect(1:num_points-1)
        for j=collect(i+1:num_points)
            #push!(G_weights, Snn_cent[i,j])
            if Snn_cent[i,j] > snn_cut_point
                if ~add_edge!(G, i, j)
                    println("Error: Cannot add edge")
                end
            end
        end
    end
    vLN, conv_history = label_propagation(G, maxiter=1000);
    # getting the corepoint labels
    # find positions in sampled_data that correpond to the gathered corepoints
    println("Number of groups detected with retrieved data:",length(unique(vLN)))
    corepoint_labels = vLN[find(x->x in overall_sample_corepoints, sampled_data)];
    #assert(length(corepoint_labels) == length(overall_sample_corepoints))
    
    results["sampled_data_snn"] = Snn_cent;
    
    results["centralized_worker_corepoints_labels"] = corepoint_labels;
    
    println("Retransmitting overall corepoints (centrally computed)...") 
    
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
            overall_sample_corepoints,
            corepoint_labels,
            inputPath,
            k = worker_result[node_id]["k"],
            KNN = KNN,
            similarity=similarity);
        end
    end
    
    #=
    ### UNDER TESTING
    #### THIS PART IS NOT WORKING...MAYBE FINDING CLUSTERS FROM SAMPLED OR COREPOINTS BASED ON SNN IS NOT A GOOD IDEA!
    #println("Overall sample size:",size(overall_sample))
    M, dim = get_header_from_input_file(inputPath);
    D = zeros(dim,length(overall_sample));
    get_slice_from_input_file(D, inputPath, overall_sample)
    
    ####
        # Apply SNN-clustering over the centralized points
        # upper bound of Eps range must be set to K
        P = tuned_snn_clustering(D, similarity=similarity, Eps_range = collect(10.0:2.0:K), MinPts_range = collect(2:5:20));

        corepoints_ix = P["corepoints"];
        cluster_labels = P["clusters"];
        cluster_assignment = P["cluster_assignment"];


        println("Retransmitting overall corepoints (centrally computed)...") 

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
                overall_sample[corepoints_ix],
                cluster_assignment[corepoints_ix],
                inputPath,
                k = worker_result[node_id]["k"],
                KNN = 1,
                similarity=similarity);
            end
        end
    =#
    
    
    overall_labels = zeros(Int64, N);
    for i=keys(final_result)
        worker_assignment = find(x -> x==i, partition);
        @assert length(worker_assignment) == length(final_result[i])
        overall_labels[worker_assignment] = final_result[i];
    end
        
    println("Joining worker's final labelings...") 
    #results["corepoints"] = overall_sample[corepoints_ix];
    results["assignments"] = overall_labels;
end

#end