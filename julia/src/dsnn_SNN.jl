module DSNN_SNN


const UNCLASSIFIED = -1;
const NOISE = -2;
const CORE_POINT = 1;
const NOT_CORE_POINT = 0;


function get_epsilon_neighbours(index::Int64, num_points::Int64, epsilon::Float64, Smat::SparseMatrixCSC{Float64, Int64})
    epsilon_neighbours_t = Int64[];
    ## sparse version
    nnz_neighbors = Smat[:, index].nzind;
    nnz_sims = Smat[:, index].nzval;
    for i in eachindex(nnz_neighbors)
        if i != index && nnz_sims[i] >= epsilon
            push!(epsilon_neighbours_t, nnz_neighbors[i])
        end
    end

    return epsilon_neighbours_t;
end

function dbscan(num_points::Int64, epsilon::Float64, minpts::Int64, Smat::SparseMatrixCSC{Float64,Int64}, point_cluster_id::Dict{Int64, Int64}, corepoints::Array{Int64,1})
    cluster_id = 1;
    for i=collect(1:num_points)
        if point_cluster_id[i] == UNCLASSIFIED
            if expand(i, cluster_id, Smat, num_points, epsilon, minpts, point_cluster_id) == CORE_POINT
                #println("Point ",i," identified as corepoint.")
                push!(corepoints, i);
                cluster_id += 1;
            end
        end
    end
end

function expand(index::Int64, cluster_id::Int64, Smat::SparseMatrixCSC{Float64,Int64}, num_points::Int64, epsilon::Float64, minpts::Int64, point_cluster_id::Dict{Int64, Int64})

    return_value = NOT_CORE_POINT;

    seeds = get_epsilon_neighbours(index, num_points, epsilon, Smat);
    num_epsilon_neighbors = length(seeds);

    #assert(num_epsilon_neighbors > 0)

    if num_epsilon_neighbors < minpts
        point_cluster_id[index] = NOISE;
        #println("Point ",index," marked as noise")
    else
        point_cluster_id[index] = cluster_id;
        #<Add epsilon neighbours to the same cluster>
        for j=collect(1:num_epsilon_neighbors)
            point_cluster_id[seeds[j]] = cluster_id;
        end
        #<See how far the cluster spreads>
        j = 1;
        while true
            # eventually, seeds array is enlarged within the spread function!
            spread(seeds[j], seeds, cluster_id, Smat, num_points, epsilon, minpts, point_cluster_id)
            if j == length(seeds)
                break
            end
            j = j + 1
        end
    
        return_value = CORE_POINT;        
    end

    return return_value;
end

function spread(index::Int64, seeds::Array{Int64,1}, cluster_id::Int64, Smat::SparseMatrixCSC{Float64,Int64}, num_points::Int64, epsilon::Float64, minpts::Int64, point_cluster_id::Dict{Int64, Int64})

    spread_neighbors = get_epsilon_neighbours(index, num_points, epsilon, Smat);
    num_spread_neighbors = length(spread_neighbors);

    #assert(num_spread_neighbors > 0);
    #<Process epsilon neighbours of neighbour>
    if num_spread_neighbors >= minpts
        for i=collect(1:num_spread_neighbors)
            d = spread_neighbors[i];
            if point_cluster_id[d] == NOISE || point_cluster_id[d] == UNCLASSIFIED
                if point_cluster_id[d] == UNCLASSIFIED
                    push!(seeds, d)
                end
                point_cluster_id[d] = cluster_id;
            end
        end
    end
end


"""
    snn_clustering(epsilon, MinPts, shared_nn_matrix)

Performs DBScan over the data whose similarity structure is represented into the _Shared Nearest Neighbor Matrix_.
"""
function snn_clustering(Eps::Float64, MinPts::Int64, Snn::SparseMatrixCSC{Float64,Int64})
    num_points = size(Snn,1)
    d_point_cluster_id = Dict{Int64, Int64}();            
    cluster_assignment = fill(UNCLASSIFIED, num_points);
    corepoints = Int64[];
    for i=collect(1:num_points)
        d_point_cluster_id[i] = UNCLASSIFIED;
    end

    dbscan(num_points, Eps, MinPts, Snn, d_point_cluster_id, corepoints)

    I = Int64[];
    J = Int64[];
    V = Int64[];

    clusters = Int64[];
    cl_label_to_col = Dict{Int64, Int64}();
    nxt_col_val = 0;

    for i=collect(1:num_points)
        #cluster_assignment[i] = d_point_cluster_id[i]
        curr_label = d_point_cluster_id[i];
        if !haskey(cl_label_to_col, curr_label)
            nxt_col_val += 1;
            push!(clusters, curr_label);
            cl_label_to_col[curr_label] = nxt_col_val;
        end
        push!(I, i);
        push!(J, cl_label_to_col[curr_label]);
        push!(V, 1);
    end 

    membership = sparse(I,J,V, num_points, nxt_col_val);

    return Dict{String, Array{Int64, 1}}("labels" => membership, "corepoints" => corepoints, "clusters" => clusters)
end



#===========================================================================================
                            THIS IS ANOTHER IMPLEMENTATION
===========================================================================================#
"""
get_corepoints(epsilon, MinPts, shared_nn_similarity)

Computes the density of each point and then identifies the corepoints.
Returns tow arrays: One with the corepoints (their index numbers) and another of length equals to the number
of points that contains a false value when the point is not a corepoint and a true value in the opposite case.
"""
function get_corepoints(Eps::Float64, MinPts::Int64, Snn::SparseMatrixCSC{Float64,Int64})
    # STEP 4 & 5 - find the SNN density of each point and identify corepoints
    N = size(Snn,1);

    snndensity = fill(0, N);
    iscore = fill(false, N);
    corepts = Int64[];
    
    for i in collect(1:N)
        for simv in Snn[:,i].nzval
            if simv  >= Eps
                snndensity[i] += 1;
            end
        end
        if snndensity[i] >= MinPts
            push!(corepts, i);
            iscore[i] = true;
        end
    end
    return corepts, iscore;
end

function get_cluster_labels(corepoints::Array{Int64, 1}, Eps::Float64, MinPts::Int64, Snn::SparseMatrixCSC{Float64,Int64})
    N = size(Snn,1);

    labels = fill(UNCLASSIFIED, N);
    isvisited = fill(false, N);
    
    current_cluster_label = 0;
    for i in eachindex(corepoints)
        p = corepoints[i];
        if isvisited[p]
            continue
        end
        isvisited[p] = true;
        current_cluster_label += 1;
        labels[p] = current_cluster_label;
        p_corenbrs = find_core_nbrs(p, corepoints, Eps, Snn);
        expand_cluster(labels, p_corenbrs, corepoints, current_cluster_label, Eps, isvisited, Snn);
    end
    return labels
end

function expand_cluster(labels::Array{Int64, 1}, corenbrs::Set{Int64}, corepoints::Array{Int64, 1}, current_cluster_label::Int64, Eps::Float64, isvisited::Array{Bool, 1}, Snn::SparseMatrixCSC{Float64,Int64})
    
    N = size(Snn,1);
    
    while length(corenbrs) > 0
        q = pop!(corenbrs);
        if isvisited[q]
            continue
        end
        labels[q] = current_cluster_label;
        isvisited[q] = true;
        q_corenbrs = find_core_nbrs(q, corepoints, Eps, Snn);
        corenbrs = union(corenbrs, q_corenbrs);
    end    
end

function find_core_nbrs(p::Int64, corepoints::Array{Int64, 1}, Eps::Float64, Snn::SparseMatrixCSC{Float64,Int64})
    N = size(Snn,1);
    corenbrs = Set{Int64}();
    for i in eachindex(corepoints)
        q = corepoints[i];
        if q != p && Snn[q,p] >= Eps
            push!(corenbrs, q);
        end
    end
    return corenbrs
end

function snn_clustering_esk03(Eps::Float64, MinPts::Int64, Snn::SparseMatrixCSC{Float64,Int64})
    corepoints, iscore = get_corepoints(Eps, MinPts, Snn);
    labels = get_cluster_labels(corepoints, Eps, MinPts, Snn);
    process_non_corepoints(labels, corepoints, iscore, Eps, Snn);
    return Dict{String, Array{Int64, 1}}("labels"=>labels, "corepoints" => corepoints)
end

function process_non_corepoints(labels::Array{Int64, 1}, corepoints::Array{Int64, 1}, iscore::Array{Bool, 1}, Eps::Float64, Snn::SparseMatrixCSC{Float64,Int64})
    N = size(Snn,1);
    
    # visit each non-corepoint
    for i in collect(1:N)
        if iscore[i]
            continue
        end
        # get the nearest corepoint
        nst_core = -1;
        nst_core_sim = -1;
        for q in Snn[:,i].nzind
            if iscore[q]  # consider only corepoint neighbors
                if Snn[q,i] > nst_core_sim
                    nst_core = q;
                    nst_core_sim = Snn[q,i];
                end
            end
        end
        if nst_core_sim >= Eps
            labels[i] = labels[nst_core];
        else
            labels[i] = NOISE;
        end
    end
end
    
end
