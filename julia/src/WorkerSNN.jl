function euclidean_sim(X::Array{Float64,2}, S::Array{Float64,2})
    # X : Data matrix of dxN
    # S : Data matrix of NxN that will be filled with cosine similarity

    N = size(X,2)
    D = pairwise(Euclidean(), X);
    
    
    tmpS = 1./(1+D);
    
    for i=collect(1:N)
        S[:,i] = tmpS[:,i];
    end
end

function cosine_sim(X::Array{Float64,2}, S::Array{Float64,2})
    # X : Data matrix of dxN
    # S : Data matrix of NxN that will be filled with cosine similarity
    
    N = size(X,2)
    norms = zeros(N)
    
    for i = collect(1:N)
        norms[i] = norm(X[:,i]);
    end
    
    newX = *(X, diagm(1./norms));
    tmpS = *(transpose(newX), newX)  
    for i=collect(1:N)
        S[:,i] = tmpS[:,i];
    end
end

function shared_nn_sim(P::Array{Float64,2}, k::Int64, Snn::Array{Float64,2}, S::Array{Float64,2};similarity::String="cosine")
    # P : Data matrix of dxN
    # k : Size of the (near)neighborhood
    # Snn is a NxN matrix and it will be filled with SNN similarity (diagonal set to 0)
    # S is a NxN matrix and it will be filled with Cosine/Euclidean similarity
    
    N= size(P,2);
    
    if similarity == "cosine"
        cosine_sim(P, S);
    elseif similarity == "euclidean"
        #println("Using euclidean similarity")
        euclidean_sim(P, S);
    else
        println("Selected similarity function not available. Using cosine function instead!") 
        cosine_sim(P, S);
    end
    
    
    nn = zeros(size(S)) #binary adjacency matrix (1 for n-neighbors)
    
    ###
    #assert((k+1) < N)
    
    for i = collect(1:N)
        nn_ixs_i = sortperm(S[:,i], rev=true)[1:k+1]    
        nn[nn_ixs_i, i] = 1;
        nn[i, i] = 0;
    end
    
    tmpS = *(transpose(nn),nn);
    #return tmpS
    for i=collect(1:N)
        Snn[:,i] = tmpS[:,i];
        Snn[i,i] = 0;
    end
        
end




function tuned_snn_clustering(D::Array{Float64,2} ;similarity::String="cosine")
    # D : Data matrix of dxN
    max_sil = -1;
    max_sil_params = ();
    k_range = collect(10:5:50);
    Eps_range = collect(1:5:30);
    MinPts_range = collect(1:5:30)
    
    
    for k = k_range
        # for a value for k ...
        #shared_nn_sim(D, k, Snn, S, similarity=similarity);
        for Eps = Eps_range
            for MinPts = MinPts_range
                #println("Eps:",Eps," MinPts:", MinPts," k:", k)
                cluster_assignment, core_points, cluster_labels, noise_found, S = snn_clustering(D, Eps, MinPts, k, similarity=similarity)
                if length(core_points) > 0
                    sil_val = mean(silhouettes(cluster_assignment, counts(cluster_assignment,maximum(cluster_assignment)), k-S));
                    
                    if sil_val > max_sil
                        max_sil = sil_val
                        max_sil_params = (cluster_assignment, core_points, cluster_labels, noise_found, Eps, MinPts, k, sil_val)
                    end
                end
             end #end of MinPts iter.
        end #end of Eps iter. 
    end

    return Dict("cluster_assignment" => max_sil_params[1], "core_points" => max_sil_params[2], "cluster_labels" => max_sil_params[3], "noise_found" => max_sil_params[4], "Eps"=>max_sil_params[5], "MinPts"=>max_sil_params[6], "k"=>max_sil_params[7], "silhouette"=>max_sil_params[8])
end




function snn_clustering(D::Array{Float64,2}, Eps::Int64, MinPts::Int64, k::Int64;similarity::String="cosine")
    # D : Data matrix of dxN
    # Initially all points have a cluster id equal to 0
    # Non noise points will be marked with labels from 1 to max_clusters - 1
    # Noise points will be marked with cluster id equal to the maximum cluster label
    # Returns cluster_assignment, core-point-instances-ids, cluster_labels, noise-detected, S
    N = size(D,2)
    
    density = zeros(Int64, N);
    cluster_assignment = zeros(Int64, N);
    cluster_labels = Int64[]
    
    Snn = Array{Float64}(N,N);    
    S = Array{Float64}(N,N);
    
    shared_nn_sim(D, k, Snn, S, similarity=similarity);
    
    #compute density
    for i = collect(1:N)
        dense_ng = find(x -> x>=Eps, Snn[:,i]);
        density[i] = length(dense_ng);
    end
    
    #identify core-points
    cpoints = find(x -> x>=MinPts, density);
    #check a non-empty 'cpoints'
    nCp = length(cpoints);

    if nCp == 0
        #println("Warning! Number of core-points equals 0.")
        return (cluster_assignment, cpoints, cluster_labels, false, S)
    end
    
    clst_id = 0;
    for i = collect(1:nCp) 
        # assign a label to each unlabeled core-points
        if cluster_assignment[cpoints[i]] == 0
            clst_id = clst_id + 1;
            push!(cluster_labels, clst_id);
            cluster_assignment[cpoints[i]] = clst_id;
        end

        # Find non labeled core points having SNN-sim > Eps
        # with current core-point and assign the same label.
        # Note: Unlabeled core-points are cpoints[unlabeled_cpoints_ixs]            
      
        #Code based on a bucle (copied from the Python version)
        for j = collect((i+1):nCp) 
            if Snn[cpoints[j], cpoints[i]] >= Eps
                cluster_assignment[cpoints[j]] = cluster_assignment[cpoints[i]];
            end
        end           
    end
    
    # Check that all core-points were assigned to a cluster (itself or to the closest)
    #assert(length(cpoints) == length(find(x -> x>0, cluster_assignment[cpoints])))

    # Mark noise points
    # Assign all non-noise, non-core points to clusters
    noncpoints = find(x -> x < MinPts, density);
    num_noncp = length(noncpoints);
    
    noise_label_marked = false
    noise_label = length(cluster_labels) + 1;
    for i=noncpoints
        # find nearest core-point
        nst_cp = cpoints[1];
        for cp=cpoints
            if Snn[cp, i] > Snn[nst_cp, i]
                nst_cp = cp;
            end
        end
        
        if Snn[nst_cp, i] < Eps
            #noise
            cluster_assignment[i] = noise_label
            if !noise_label_marked
                push!(cluster_labels, noise_label);
                noise_label_marked = true;
            end
        else
            cluster_assignment[i] = cluster_assignment[nst_cp] 
        end
        ##
        
    end

    # Check that all points were assigned to a cluster (noise included)
    assert(length(cluster_assignment) == length(find(x -> x > 0, cluster_assignment)) )

    return (cluster_assignment, cpoints, cluster_labels, noise_label_marked, Snn)
end



##################################################################################################################
################################################## PARALLEL METHODS ##############################################
##################################################################################################################

function adjust_weight(inputVec::Array{Float64,1}; gm=0.2, r=2)
    #alpha_gm = 0.5*log((1-gm)/gm)-0.5*(log((1-gm)/gm))^(1/r);
    #return map((x) -> (x<gm)?(0.5*(log((1-x)/x))^(1/r) + alpha_gm):(0.5*log((1-x)/x)), inputVec)
    return map((x) -> exp(-x*x), inputVec)
    #return map((x) -> x, inputVec)
end



function local_work(assigned_instances::Array{Int64,1}, inputPath::String, Eps::Int64, MinPts::Int64, k::Int64, pct_sample::Float64;similarity::String="cosine")
    #=
     Takes a file path where the data matrix is stored.
    
    It is inmportant that assigned_instances is sorted in ascending order as in this way the indexes will match the 
    column ids of the matrix retrieved by get_slice_from_input_file.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#

    M, dim = get_header_from_input_file(inputPath)
    #d = Array{Float64,2}(dim, length(assigned_instances));
    d = zeros(dim, length(assigned_instances));
    get_slice_from_input_file(d, inputPath, assigned_instances);
    
    # Coarse-grained parameter search guided by silhouette.
    # silhouettes(convert(Vector{Int64}, cluster_assignment_m), counts(cluster_assignment_m), 1-Snn)
    #cluster_assignment, core_points, cluster_labels = snn_clustering(d, Eps, MinPts, k, similarity=similarity)
    cluster_assignment, core_points, cluster_labels, (sil_value, t_Eps, t_MinPts, t_k) = tuned_snn_clustering(d, similarity=similarity)
    
    
    # Note that the indexes appearing within arrays cluster_assignment and core_points are in range [1:lenght(assigned_instances)]
    # ... thus in order to obtain the index relative to the array A (whole data), the indirection to apply is 
    # ... assigned_instances[i]
    ###

    nCp = length(core_points)
    n_sampled_points = round(Int, pct_sample*nCp);
    
    if length(cluster_labels) == 0
        return []
    end
    
    corepts_labels = cluster_assignment[core_points];# same length as core_points filled with their cluster labels
    data_weight = zeros(nCp);
    
    Ns = length(find(x -> x>0, cluster_assignment));#Nr of labeled points
    
    for i = cluster_labels
        cpts_in_cluster_i = find(x -> x==i, corepts_labels);#relative to vector  core_points
        in_cluster_i = find(x -> x==i, cluster_assignment);#Nr of labeled pts in cluster i
        Nc = length(in_cluster_i);
        #Nc = length(cpts_in_cluster_i);
        #Wc = 1e-10 + ( (1.0 - (Nc/Ns)) / (2.0*Nc) ); # 1e-10 added to avoid problems when only one cluster is found.
        #Wc = 1e-20 + (Ns/Nc); # 1e-10 added to avoid problems when only one cluster is found.
        Wc = Nc/Ns;
        data_weight[cpts_in_cluster_i] = Wc;
    end
    #data_weight = adjust_weight(data_weight/sum(data_weight), gm=0.4, r=4);
    #data_weight = adjust_weight(exp(data_weight)/sum(exp(data_weight)), gm=0.4, r=3);
    data_weight = adjust_weight(data_weight);
    data_weight =  exp(data_weight)/sum(exp(data_weight));
    #data_weight = data_weight/sum(data_weight);
    
    sampled_points = sample(collect(1:nCp), WeightVec(data_weight), n_sampled_points , replace=false);
    return assigned_instances[core_points[sampled_points]]
end


#=
********************************
        New version
********************************
=#

function local_work_final(assigned_instances::Array{Int64,1}, corepoints::Array{Int64,1}, corepoint_labels::Array{Int64,1}, inputPath::String;KNN::Int64=5, similarity::String="cosine")
    #=
     Takes a file path where the data matrix is stored.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#

    
    L = Int64[];
    append!(L, assigned_instances);
    for i=collect(1:length(corepoints))
        if ~ (corepoints[i] in assigned_instances)
            push!(L, corepoints[i])
        end
    end
    sort!(L)
    
    instance_map = Dict{Int64, Int64}(zip(L, collect(1:length(L)) )) # maps each instance-id to the corresponding data column id
    
    N  = length(L);
    
    M, dim = get_header_from_input_file(inputPath)
    #d = Array{Float64,2}(dim, length(assigned_instances));
    D = zeros(dim, N);
    get_slice_from_input_file(D, inputPath, L);
    
    Snn = Array{Float64}(N,N);    
    S = Array{Float64}(N,N);
    
    shared_nn_sim(D, k, Snn, S, similarity=similarity);
    
    # get the core point indexes in the similarity matrix.
    corepoint_ixs = map(x->instance_map[x], corepoints);
    
    
    assigned_instances_labels = zeros(size(assigned_instances))
    # for each non-corepoint, find its similarity to the corepoints and take the top K
    for i=collect(1:length(assigned_instances) )
        if assigned_instances[i] in corepoints
            corepoint_ix = find(x->x==assigned_instances[i], corepoints)[1];
            assigned_instances_labels[i] = corepoint_labels[corepoint_ix]
            continue
        end
        # Snn[instance_map[i],:] contains the similarity with the remaining points.
        instance_ix = instance_map[assigned_instances[i]]; # index in the similarity matrix
        # Get the label majority from the K nearest corepoints. 
        core_sims = Snn[instance_ix, corepoint_ixs];
        knncorepoints_rank = sortperm(core_sims, rev=true)[1:KNN]; # sort from max to min and get the top KNN
        # Each item in 'knncorepoints_rank' denotes the similarity of the correponding item index in the corepoints and corepoint_labels arrays.
        voter_labels = corepoint_labels[knncorepoints_rank];
        assigned_instances_labels[i] = span(voter_labels)[sortperm(counts(voter_labels), rev=true)[1]] #returns the most common label.
    end
    return assigned_instances_labels
end
