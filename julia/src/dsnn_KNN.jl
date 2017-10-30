module DSNN_KNN

type ApIndexJoin
    idx :: Array{Array{Tuple{Float64,Int64}}}; # feature_i -> [(value_of_feature_i, doc_id)...]
    num_features :: Int64;
    num_instances :: Int64;
    #object_norms :: Array{Float64,1}; #assumes that al vectors have unit norm
    epsilon :: Float64;
    H :: Dict{Int64, Dict{Int64, Float64}}; # dict for features and then documents with their suffix norms.
    nbrs :: Dict{Int64, Array{ Tuple{Float64, Int64} }}; # dict with the k neighbors for each object
    ApIndexJoin() = new()
end

function init_structure!(n_instances::Int64, n_features::Int64, o::ApIndexJoin)
    o.idx = Array{Tuple{Float64, Int64}}[]; # List of tuple-arrays
    for i = collect(1:n_features)
        push!(o.idx, Tuple{Float64, Int64}[])
    end
    o.num_features = n_features;
    o.num_instances = n_instances;
    #o.object_norms = Float64[];
    o.nbrs = Dict{Int64, Array{ Tuple{Float64, Int64} }}();
    o.H = Dict{Int64, Dict{Int64, Float64}}();
end

function suffix_norm(o::SparseVector{Float64,Int64}, p)
    """
    returns the p-suffix norm. This is the norm of the vector by only considering
    the values of its (|o| - p) trailing features.
    """
    #assert( p < length(o))
    #return sqrt(sum(o[(p+1):end].^2))
    #= Esta optimiz. funciona lento !!
    if p > o.nzind[end]
        return 0.0
    else
        # k is the next nnz feature higher than p
        k = o.nzind[find(x-> x > p, o.nzind)[1]];
        o_nnz_part = o[o.nzind[k:end]];
        return sqrt(dot(o_nnz_part, o_nnz_part))
    end
    =#

    #return sqrt(dot(o[(p+1):end], o[(p+1):end]))
    return norm(o[(p+1):end])
end

function build_index!(ix::ApIndexJoin, M::SparseMatrixCSC{Float64, Int64}, epsilon::Float64; sort_lists::Bool=false)
    """
    Create the partial inverted index.
    ´ix´ is an initiaized Approximate Index .
    ´M´ is a sparse matrix containing the UNIT-NORM data vectors. Objects in columns and features in rows.
    ´epsilon´ is the similarity threshold.
    ´sort_lists´ is a boolean parameter indicating if the feature lists will be sorted in non-ascending order.
    """
    nfeatures, nobjects = size(M);

    """ columns are already normalized-
    for q=collect(1:nobjects)
        norm_v = suffix_norm(M[:,q], 0);
        for qi = M[:,q].nzind
            M[qi,q] = M[qi,q] / norm_v;
        end

        #push!(ix.object_norms, norm_v); --> all objects were normalized
    end
    """

    # Traverses the matrix by columns visiting each object vector.

    for q = collect(1:nobjects)
        v = M[:,q];

        for j = collect(1:(nnz(v)-1))
            feature = v.nzind[j]; # current non zero feature in the vector.
            suffix_norm_feat = suffix_norm(v, feature);
            # Store the feature and the object suffix norm.
            if !haskey(ix.H, feature)
                ix.H[feature] = Dict{Int64, Float64}()
            end
            ix.H[feature][q] = suffix_norm_feat;

            if suffix_norm_feat >= epsilon
                # index the vector onto its feature list.
                push!(ix.idx[feature], (v[feature], q));
            else:
                break
            end

        end
    end

    if sort_lists
        for i = collect(1:length(ix.idx))
            if length(ix.idx[i]) > 1
                sort!(ix.idx[i], rev=true);
            end
        end
    end
end

function bounded_sim(q::Int64, c::Int64, ix::ApIndexJoin, M::SparseMatrixCSC{Float64, Int64}, epsilon::Float64)
    s = 0.0;
    dq = M[:,q];
    dc = M[:,c];

    #for j = collect(1:ix.nfeatures)
    #    if ix.idx[j]
    #end

    for j in dc.nzind
        if dq[j] > 0
            s = s + dq[j]*dc[j]
            if !haskey(ix.H, j)
               ix.H[j] = Dict{Int64, Float64}()            
            end
            if !haskey(ix.H[j], q)
                #compute suffix norm for dq and store it
                suffix_norm_feat = suffix_norm(dq, j);
                ix.H[j][q] = suffix_norm_feat;
            end
            if !haskey(ix.H[j], c)
                #compute suffix norm for dq and store it
                suffix_norm_feat = suffix_norm(dc, j);
                ix.H[j][c] = suffix_norm_feat;
            end

            if (s + ix.H[j][q]*ix.H[j][c]) < epsilon
                return -1.0
            end
        end
    end

    return s
end

function fill_index!(ix::ApIndexJoin, M::SparseMatrixCSC{Float64, Int64}, k::Int64, epsilon::Float64, mu_1::Int64)
    """
    Vector (columns) in ´M´ are normalized and index ´ix´ is built.
    """
    build_index!(ix, M, epsilon, sort_lists=true);

    nobjects = size(M, 2);
    #T = zeros(Bool, nobjects); # accounts for the processed objects.
    # Not here! L = Tuple{Float64, Int64}[]; # candidate list

    for q = collect(1:nobjects)
        L = Tuple{Float64, Int64}[]; # candidate list
        T = zeros(Bool, nobjects);   

        if !haskey(ix.nbrs, q)
            ix.nbrs[q] = Tuple{Float64, Int64}[];
        end
        for (s_cq, dc) in ix.nbrs[q]
            T[dc] = true;
        end
        l = 0;

        # RANK nnz features by weight
        v = M[:, q];
        feat_pq = Collections.PriorityQueue(Int64, Float64, Base.Order.Reverse); # Forward implies lesser to higher        
        for i in v.nzind
            feat_pq[i] = v[i]
        end

        # traverse the Feature-Index by following the RANK
        while length(feat_pq) > 0
            j, q_j = Collections.peek(feat_pq); # getting the next feature for processing nghbs
            Collections.dequeue!(feat_pq);

            # Recall that ix.idx[j] is a tuple-array of points containing that feature (sorted by weight)
            # Start checking Points with highest weight within the Array.
            for c = collect(1:length(ix.idx[j])) 
                dc_j, dc = ix.idx[j][c];

                if dc > q && !T[dc]
                    s = bounded_sim(q, dc, ix, M, epsilon);
                    if s >= epsilon
                        push!(L, (s, dc));                        
                        if !haskey(ix.nbrs, dc)
                            ix.nbrs[dc] = Tuple{Float64, Int64}[];
                        end
                        if length(ix.nbrs[dc]) < k
                            push!(ix.nbrs[dc], (s, q));
                        end
                    end
                    T[dc] = true;
                end
                l = l + 1;

                if l == mu_1
                    break
                end
            end

            if l == mu_1
                break
            end
        end
        # Add current neighbors from Nq to L
        for (s_cq, dc) in ix.nbrs[q]
            push!(L, (s_cq, dc));
            #L[dc] = s_cq;
        end
        # Sort L and put top-k pairs into Ni[q]
        resize!(ix.nbrs[q], 0);
        sort!(L, rev=true); # sorting candidate points by similarity against q
        max_len = length(L);

        ix.nbrs[q] = L[1:min(k,max_len)]; # Nq set to top-k points
    end
end

function improve_graph!(ix::ApIndexJoin, M::SparseMatrixCSC{Float64, Int64}, k::Int64, epsilon::Float64, mu_2::Int64)

    nfeatures, nobjects = size(M);

    # build reverse-neighborhood            
    I = Dict{Int64, Array{ Tuple{Float64, Int64} }}()
    for q = collect(1:nobjects)
        I[q] = Tuple{Float64, Int64}[];
    end
    for q in keys(ix.nbrs)
        for (sval, nbr) in ix.nbrs[q]
            # add q to the nbr index.
            push!(I[nbr], (sval, q))
        end
    end
    for q = collect(1:nobjects)
        L = Tuple{Float64, Int64}[]; # candidate list
        T = zeros(Bool, nobjects);
        inCList = zeros(Bool, nobjects);
        Q = Collections.PriorityQueue(Int64, Float64, Base.Order.Reverse)
        l = 0;
        for (s_cq, dc) in I[q] # tagging all points whos nbrhds contain point q
            T[dc] = true;
            push!(L, (s_cq, dc));
            inCList[dc] = true;
        end
        for (s_cq, dc) in ix.nbrs[q]
            T[dc] = true;
            Q[dc] = s_cq; # queueing neighbors (sorting by similarity)
        end
        while length(Q) > 0
            dc, s_cq = Collections.peek(Q); # getting the next most similar neighbor
            Collections.dequeue!(Q);
            if !inCList[dc]
                push!(L, (s_cq, dc));
                inCList[dc] = true
            end
            if l < mu_2
                for (s_cv, dv) in ix.nbrs[dc]
                    if !T[dc]
                        #s = bounded_sim(q, dc, ix, M, epsilon);
                        s_qv = bounded_sim(q, dv, ix, M, epsilon);
                        if s_qv >= epsilon
                            Q[dv] = s_qv; # queueing in max_heap (sorted by similarity)
                        end
                        T[dc] = true;#chequear!
                    end
                end
            end
            l = l + 1;
        end

        # check duplicates
        sort!(L, rev=true);
        for i = collect(1:length(L)-1)
            if L[i][2] == L[i+1][2]
                print("Duplicate entry found (",L[i+1][2],")")
            end
        end
        ##
        for (s_cq, dc) in ix.nbrs[q]
            if !inCList[dc]
                push!(L, (s_cq, dc));
            end
        end
        # Sort L and put top-k pairs into Ni[q]
        resize!(ix.nbrs[q], 0);
        sort!(L, rev=true);
        max_len = length(L);
        #println("LEN(L,",q,"):",max_len)
        ix.nbrs[q] = L[1:min(k,max_len)];
    end
end

function initialAppGraph(M::SparseMatrixCSC{Float64,Int64}, k::Int64, epsilon::Float64, mu_1::Int64)
    """
    InitialGraph method from the paper.

    Builds the approximate e-Index for KNN computation. The sparse matrix `M` 
    containes objects in its Columns and features in its Rows.
    """
    nfeatures, nobjects = size(M);

    ap_ix = ApIndexJoin();
    init_structure!(nobjects, nfeatures, ap_ix);

    fill_index!(ap_ix, M, k, epsilon, mu_1);

    return ap_ix
end


"""
    get_snnmatrix(ApproxSimIndex, KNN)

Returns the shared-nn similarity matrix.
Employs an instance of ApIndexJoin to estimate pairwise distances.
"""
function get_knnmatrix(ix::ApIndexJoin, KNN::Int64)
    
    # 1st. The near neighbor adjaceny matrix is built.
    # consider that the near neighbors for object i are represented by the
    # rows having a 1 in column i
    I = Int64[];
    J = Int64[];
    V = Int8[];
    for q_i in collect(1:ix.num_instances)
        maxnn_len = min(KNN, length(ix.nbrs[q_i]));   
        
        for nninfo in ix.nbrs[q_i][1:maxnn_len]
            nn_ix = nninfo[2];
            push!(J, q_i); #column denoting the current object
            push!(I, nn_ix); # row denoting the near neighbor
            push!(V, Int8(1));
        end
    end
    nnmat = sparse(I,J,V, ix.num_instances, ix.num_instances);
    return nnmat;
end

function get_snnmatrix(ix::ApIndexJoin, KNN::Int64)
    """
    Returns the shared-nn similarity matrix.
    """
    nnmat = get_knnmatrix(ix, KNN);
    
    snnmat = *(transpose(nnmat),nnmat) / KNN;
    return snnmat;
end



end
