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

    return Dict{String, Any}("labels" => membership, "corepoints" => corepoints, "clusters" => clusters)
end


using DataStructures

"""
    get_dataclusters_sample(epsilon, minpoints, SNN_matrix, sampling_percentage)

Identifies corepoints and select the top nbrs of each one (a sampling_percentage of its nbrhd).
"""
function get_dataclusters_sample(Eps::Float64, MinPts::Int64, Snn::SparseMatrixCSC{Float64,Int64}, pct_sample::Float64)
    #corepoints, iscore = get_corepoints(Eps, MinPts, Snn);
    N = size(Snn,1);

    iscore = fill(false, N);
    corepts = Int64[];
    sample = Int64[];
    
    for i in collect(1:N)
        snndensity = 0;
        pq_i = DataStructures.PriorityQueue{Int64, Float64}(Base.Order.Reverse); # higher to lower sims
        for e_nbr in Snn[:,i].nzind # nnz nbrs of point i
            if e_nbr == i
                continue
            end
            simv = Snn[e_nbr,i];
            if simv  >= Eps
                pq_i[e_nbr] = simv
            end
        end
        
        if length(pq_i) >= MinPts
            push!(corepts, i);
            iscore[i] = true;
            # get %pct_sample of its nbrhd starting from the most similar ones.
            knn_chosen = round(Int64, pct_sample * length(pq_i));
            chosen_cnt = 0;
            while length(pq_i) > 0 && chosen_cnt < knn_chosen
                j, sim_ij = DataStructures.peek(pq_i);
                DataStructures.dequeue!(pq_i);    
                push!(sample, j);
                chosen_cnt += 1;
            end
            #generate sample of nbrs
            #sample_C = Stats.sample(RNG_W, cl_labels[:,C].nzind, ceil(Int64, C_size*pct_sample), replace=false);
        end
    end
    s_sample = Set(sample); # eliminates the duplicates
    sample = sort(collect(s_sample));# sort the points
    return corepts, sample;
end
end
