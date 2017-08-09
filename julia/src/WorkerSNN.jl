# 03.08.2017
#module WorkerSNN
function silhouette(D::Array{Float64,2}, labels::Array{Int64,1})
    #D: Distance matrix
    #labels: vector with assignments
    #
    # The best value is 1 and the worst value is -1. 
    # Values near 0 indicate overlapping clusters. 
    # Negative values generally indicate that a sample has been assigned to the wrong cluster
    #
    n = size(D)[1]
    sil_sum = 0.0
    for i=collect(1:n)
        sil_sum += silhouette_i(i, D, labels)
    end
    return sil_sum / n
end
    
function silhouette_i(i::Int64, D::Array{Float64,2}, labels::Array{Int64,1})    
    #i: current point to examine
    #D: Distance matrix
    #labels: vector with assignments
    #
    # Compute the Silhouette Coefficient for a specific sample.
    #
    A = labels[i]
    points_in_A = find(x->x==A, labels)
    a_i = (sum(D[i, points_in_A])-D[i,i])/(size(points_in_A)[1] - 1)#It is assumed that D[i,i]:=0
    b_i = Inf
    for c=unique(labels) #computing min ave.dist among i and items in other clusters.
        if c == A
            continue
        end
        points_in_c = find(x->x==c, labels)
        ave_dist_i = sum(D[i, points_in_c])/size(points_in_c)[1]
        if ave_dist_i < b_i
            b_i = ave_dist_i
        end
    end
    return (b_i - a_i)/(max(b_i, a_i))
end

function cvnn_index(D::Array{Float64,2}, labels::Array{Int64,1}; sep_k::Int64=10)
    #D: Distance matrix
    #labels: Array with one class label for each point in D
    #sep_k: Size of the neighborhoods
    sep_score = 0.0
    com_score = 0.0
    for c=unique(labels)
        points_in_c = find(x->x==c, labels)
        n_c = size(points_in_c)[1]

        sum_c = 0.0
        for j=points_in_c
            knn_j = sortperm(D[:,j])[2:(sep_k + 1)] #k-nst-n (ascending order in dist)
            q_j = size(find(x->x!=c, labels[knn_j]))[1] #nst-n in different group
            sum_c += q_j/sep_k
        end
        sep_c = (1.0/n_c)*sum_c #average weight for objs in the current cluster.
        if sep_c > sep_score
            sep_score = sep_c
        end
        ##
        sum_c = 0.0
        sims_c = D[points_in_c,points_in_c]
        for i=collect(1:(n_c-1))
            for j=collect((i+1):n_c)
                sum_c += D[i,j]
            end
        end
        com_score += (2.0/(n_c*(n_c-1)))*sum_c        
    end
    return (com_score + sep_score)
end

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


function shared_nn_sim(P::Array{Float64,2}, k::Int64, Snn::Array{Float64,2}, S::Array{Float64,2})
    #=
    Computes only the SNN similarity matrix with give parameters k and the similarity matrix (already comptued)
    Returns nothing, since it modifies the Snn argument already initialized.
    
    P : Data matrix of dxN
    k : Size of the (near)neighborhood
    Snn is a NxN matrix and it will be filled with SNN similarity (diagonal set to 0)
    S is a NxN matrix with Cosine/Euclidean similarity
    =#
    N= size(P,2);
    
    nn = zeros(size(S)) #binary adjacency matrix (1 for n-neighbors)
    
    ###
    #assert((k+1) < N)
    
    k = min(k, N-1); #In order to avoid BoundErrors on line 63
    
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

function compute_similarities(P::Array{Float64,2}, k::Int64, Snn::Array{Float64,2}, S::Array{Float64,2};similarity::String="cosine")
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
    
    # now computes the snn sim matr.
    shared_nn_sim(P, k, Snn, S)            
end


function compute_similarity(P::Array{Float64,2}, S::Array{Float64,2};similarity::String="cosine")
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
end




function tuned_snn_clustering(D::Array{Float64,2} ;k_range = [40, 60, 80, 100, 150], similarity::String="cosine")
    # D : Data matrix of dxN
    max_sil = -1;
    max_sil_params = ([], [], -1.0, -1, -1, -1, [], []);

    #println("Using Dbscan (Julia)")
    num_points = size(D,2);
    S = zeros(Float64, num_points, num_points);

    compute_similarity(D, S, similarity=similarity);
    Snn = zeros(Float64, num_points, num_points);
    
    for k = k_range
    
        Eps_range = collect(10:10.0:k)
        MinPts_range = collect(10:10:k)
        shared_nn_sim(D, k, Snn, S)
        #compute_similarities(D, k, Snn, S, similarity=similarity);
        Dnn = k-Snn;
        for epsilon = Eps_range
            for MinPts = MinPts_range
                
                d_point_cluster_id = Dict{Int64, Int64}();
                cluster_assignment = fill(SNNDBSCAN.UNCLASSIFIED, num_points);
                corepoints = Int64[];
                for i=collect(1:num_points)
                    d_point_cluster_id[i] = SNNDBSCAN.UNCLASSIFIED;
                end

                SNNDBSCAN.dbscan(num_points, epsilon, MinPts, Snn, d_point_cluster_id, corepoints)
                for i=collect(1:num_points)
                    cluster_assignment[i] = d_point_cluster_id[i]
                end
                assigned = find(x-> x>0, cluster_assignment);
                
               
                if length(corepoints) > 0 && length(assigned) >= num_points*0.4
                    #sil_val = cvnn_index(k-Snn[assigned,assigned], cluster_assignment[assigned]);
                    #if sil_val < max_sil
                    #    max_sil = sil_val
                    #    clusters = unique(cluster_assignment[assigned]);
                    #    max_sil_params = (cluster_assignment, corepoints, epsilon, MinPts, k, sil_val, assigned, clusters)
                    #end

                    sil_val = silhouette(Dnn[assigned,assigned], cluster_assignment[assigned]);
                    #sil_val = mean(Clustering.silhouettes(cluster_assignment[assigned], Clustering.counts(cluster_assignment[assigned],maximum(cluster_assignment[assigned])), k-Snn[assigned,assigned]));
                    if sil_val > max_sil
                        max_sil = sil_val
                        clusters = unique(cluster_assignment[assigned]);
                        max_sil_params = (cluster_assignment, corepoints, epsilon, MinPts, k, sil_val, assigned, clusters)
                    end
                    
                end
                
             end #end of MinPts iter.
        end #end of Eps iter. 
    end
 
    if max_sil == -1
        println("No parameter setting found.");
    end
    #assert(length(max_sil_params) > 0) # otherwise, no valid clustering was achieved!
    
    return Dict("cluster_assignment" => max_sil_params[1], "corepoints" => max_sil_params[2], "Eps"=>max_sil_params[3], "MinPts"=>max_sil_params[4], "k"=>max_sil_params[5], "silhouette"=>max_sil_params[6],"assigned_points"=>max_sil_params[7], "clusters"=>max_sil_params[8])
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
    
    compute_similarities(D, k, Snn, S, similarity=similarity);
    
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



#function local_work(assigned_instances::Array{Int64,1}, inputPath::String, pct_sample::Float64;similarity::String="cosine", Eps_range = collect(5.0:5.0:50.0), MinPts_range = collect(20:10:50),k_range = [40, 50])
function local_work(assigned_instances::Array{Int64,1}, inputPath::String, pct_sample::Float64;similarity::String="cosine")
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
    
    #P =  tuned_snn_clustering(d, Eps_range=Eps_range, MinPts_range=MinPts_range, k_range=k_range, similarity=similarity);
    P =  tuned_snn_clustering(d,similarity=similarity);
    #cluster_assignment, core_points, cluster_labels, (sil_value, t_Eps, t_MinPts, t_k)
    
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



function local_work_final(assigned_instances::Array{Int64,1}, corepoints::Array{Int64,1}, corepoint_labels::Array{Int64,1}, inputPath::String;k::Int64 = 50, KNN::Int64=5, similarity::String="cosine")
    #=
     Takes a file path where the data matrix is stored.
    
     Returns a core-point sample (col ids of the original matrix as denoted by assigned_instances).
     This sample is built from the core-points detected by the SNN algorithm ran within the node.
    =#

    already_assigned_corepoints = Int64[];
    already_assigned_corepoint_labels = Int64[];
    
    L = Int64[];
    append!(L, assigned_instances);
    for i=collect(1:length(corepoints))
        # if corepoints was already assigned to this worker, mark it along with its label
        if corepoints[i] in assigned_instances
            corepoint_index_in_assigned_instances = find(x->x==corepoints[i], assigned_instances)
            push!(already_assigned_corepoints, corepoint_index_in_assigned_instances[1])
            push!(already_assigned_corepoint_labels, corepoint_labels[i]) 
        else #otherwise appendit to the local (worker) data so it is included in the similarity computation.
            push!(L, corepoints[i])            
        end
    end
    sort!(L)
    
    # j is an assigned instance or a core point, then 
    # instance_map[j] -> Denotes the column number of the local Similarity matrix.
    data_to_local = Dict{Int64, Int64}(zip(L, collect(1:length(L)) )) # maps each instance-id to the corresponding data column id
    
    N  = length(L);
    
    M, dim = get_header_from_input_file(inputPath)
    #d = Array{Float64,2}(dim, length(assigned_instances));
    D = zeros(dim, N);
    get_slice_from_input_file(D, inputPath, L);
    
    Snn = Array{Float64}(N,N);    
    S = Array{Float64}(N,N);
    
    compute_similarities(D, k, Snn, S, similarity=similarity);
    
    # Identify the mapping between local data indexes and overall data indexes
    # corepoint indexes in the similarity matrix.
    corepoint_ixs = map(x->data_to_local[x], corepoints);
    # borderpoint indexes in the similarity matrix.
    assigned_without_corepoints = find(x->~(x in corepoints), assigned_instances);# positions in assigned_instances
    border_instances = assigned_instances[assigned_without_corepoints];
    borderpoint_ixs = map(x->data_to_local[x], border_instances);
    
    nst_cp_label = fill(-10, length(borderpoint_ixs));
    nst_cp_dist = fill(Inf,length(borderpoint_ixs)); #initialization with maximum distance
    
    # for each corepoint, get the shortest path to every border point.
    for cp_i=collect(1:length(corepoint_ixs))
        # get corepoint label
        curr_cp_label = corepoint_labels[cp_i];
        # get shortest distance to every point
        res = SNNGraphUtil.shortest_path(k-Snn, corepoint_ixs[cp_i]); #computes distance to every point in 1:N
        
        #check only the borderpoint_ixs
        for i=collect(1:length(borderpoint_ixs))
            #if previous shortest distance is larger than current, update the label and min distance             
            if nst_cp_dist[i] > res["dist"][borderpoint_ixs[i]]
                nst_cp_dist[i] = res["dist"][borderpoint_ixs[i]]
                nst_cp_label[i] = curr_cp_label
            end
        end
    end
    
    ####
    #assert(length(already_assigned_corepoints)+length(border_instances) == length(assigned_instances))
    
    assigned_instances_labels = zeros(size(assigned_instances))
    # Assign first the corepoints appearing IN assigned_instances
    assigned_instances_labels[already_assigned_corepoints] = already_assigned_corepoint_labels;
    # Use border_instances to assign the remaining ones
    assigned_instances_labels[assigned_without_corepoints] = nst_cp_label;
    return assigned_instances_labels
end
#end
