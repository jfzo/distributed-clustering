@everywhere function euclidean_sim(X::Array{Float64,2}, S::Array{Float64,2})
    # X : Data matrix of dxN
    # S : Data matrix of NxN that will be filled with cosine similarity

    N = size(X,2)
    D = pairwise(Euclidean(), X);
    min_D = 0.0;
    max_D = 0.0;
    for i=collect(1:N)
        v = maximum(D[:,i]);
        if v > max_D
            max_D = v;
        end
    end

    tmpS = 1 - (D - min_D)/(max_D-min_D);
    
    for i=collect(1:N)
        S[:,i] = tmpS[:,i];
    end
end

@everywhere function cosine_sim(X::Array{Float64,2}, S::Array{Float64,2})
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

@everywhere function shared_nn_sim(P::Array{Float64,2}, k::Int64, Snn::Array{Float64,2}, S::Array{Float64,2};similarity::String="cosine")
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

@everywhere function snn_clustering(D::Array{Float64,2}, Eps::Int64, MinPts::Int64, k::Int64;similarity::String="cosine")    
    # D : Data matrix of dxN

    N = size(D,2)
    
    density = zeros(Int32, N);
    cluster_assignment = zeros(Int32, N);
    cluster_labels = []
    
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
        return (cluster_assignment, cpoints, cluster_labels)
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
      
        #=
        unlabeled_cpoints = find(x -> x==0, cluster_assignment[cpoints[i+1:nCp]]); #relative to cluster_assignment[cpoints]
        unlabeled_cpoints_ixs = cpoints[i+1:nCp][unlabeled_cpoints];# relative to 1:N
        if length(unlabeled_cpoints_ixs) == 0
            break
        end
        # Note: near_cpoints are current columns of the SNN Matrix
        near_cpoints = find(x -> x>=Eps, Snn[unlabeled_cpoints_ixs, cpoints[i]]);
        cluster_assignment[unlabeled_cpoints_ixs[near_cpoints]] = clst_id;

        =#
              
        
        #Code based on a bucle (copied from the Python version)
        for j = collect((i+1):nCp) 
            if Snn[cpoints[j], cpoints[i]] >= Eps
                cluster_assignment[cpoints[j]] = cluster_assignment[cpoints[i]];
            end
        end           
    end
    
    #= 
    # This block does not have to be used, but in any case it is put 
    # to ensure the labeling of all cpoints .
    
    unlabeled_cpoints_ixs = find(x -> x==0, cluster_assignment[cpoints]);
    next_labels = collect((clst_id + 1):(clst_id + length(unlabeled_cpoints_ixs) ) );
    cluster_assignment[unlabeled_cpoints_ixs] = next_labels;
    
    =#

    # assert(length(cpoints) == length(find(x -> x>0, cluster_assignment[cpoints])))

    # Mark noise points
    # Assign all non-noise, non-core points to clusters
    noncpoints = find(x -> x<MinPts, density);
    num_noncp = length(noncpoints);
    
    for i=noncpoints
        # find nearest core-point
        #=
        #COde based on a bucle (copied from the Python version)
        ns_cpoint = cpoints[sortperm(Snn[cpoints,i], rev=true)[1] ]# nearest core-point (relative to 1:N)

        if Snn[ns_cpoint, i] < Eps
            cluster_assignment[i] = -1 #noise
        else
            cluster_assignment[i] = cluster_assignment[ns_cpoint] #non-noise
        end
        ##
        =#
        
        nst_cp = cpoints[1];
        for cp=cpoints
            if Snn[cp, i] > Snn[nst_cp, i]
                nst_cp = cp;
            end
        end
        
        if Snn[nst_cp, i] < Eps
            #noise
            cluster_assignment[i] = -1
        else
            cluster_assignment[i] = cluster_assignment[nst_cp] 
        end
        ##
        
    end

    return (cluster_assignment, cpoints, cluster_labels)
end

####
@everywhere function local_work(A::SharedArray{Float64,2}, assigned_instances::Array{Int64,1}, Eps::Int64, MinPts::Int64, k::Int64, pct_sample::Float64;similarity::String="cosine")
    # Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
    # This sample is built from the core-points detected by the SNN algorithm ran within the node.

    
    # It is assumed that 'A' contains the data matrix (rows are features and columns are instances).
    d = A[:,assigned_instances] # the assigned instance columns (d is now a normal Array)
    
    cluster_assignment, core_points, cluster_labels = snn_clustering(d, Eps, MinPts, k, similarity=similarity)
    # Note that the indexes appearing withon arrays cluster_assignment and core_points are in range [1:lenght(assigned_instances)]
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
    
    for i = cluster_labels
        in_cluster_i = find(x -> x==i, corepts_labels);#relative to vector  core_points
        Nc = length(in_cluster_i);
        Wc = 1e-10 + ( (1.0 - (Nc/nCp)) / (2.0*Nc) ); # 1e-10 added to avoid problems when only one cluster is found.
        data_weight[in_cluster_i] = Wc;
    end
    
    sampled_points = sample(collect(1:nCp), WeightVec(data_weight), n_sampled_points , replace=false);
    return assigned_instances[core_points[sampled_points]]

    #=
    
    # Sampling core-points
    # computing weights
    N = size(d,2)
    clustered_pts = find(x -> x>0, cluster_assignment);
    Ns = length(clustered_pts) # number of points having a cluster-id assigned
    
    #println("#noisy pts:",length(find(x -> x<0, cluster_assignment))," #0-label pts:",length(find(x -> x==0, cluster_assignment))," #labeled pts:",length(find(x -> x>0, cluster_assignment)) )
    
    n_sampled_points = round(Int, pct_sample*Ns);

    #println("#labeled points:", Ns ," #pin:",N," clabels:","[",cluster_labels,"]","#spnts:",n_sampled_points)

    if length(cluster_labels) == 0
        return []
    end
    
    data_weight = zeros(N)
    
    for i = cluster_labels
        in_cluster_i = find(x -> x==i, cluster_assignment);
        Nc = length(in_cluster_i); #number of points in the current cluster
        Wc = 1e-10 + ( (1.0 - (Nc/Ns)) / (2.0*Nc) ); # 1e-10 added to avoid problems when only one cluster is found.
        data_weight[in_cluster_i] = Wc;
    end
    
    #points not assigned to a cluster have 0 weight -> They won't be sampled!
    sampled_points = sample(collect(1:Ns), WeightVec(data_weight[clustered_pts]), n_sampled_points , replace=false);

    return assigned_instances[clustered_pts[sampled_points]]
    
    =#
    
end
