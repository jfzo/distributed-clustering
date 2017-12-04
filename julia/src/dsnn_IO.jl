module DSNN_IO


"""
    read_configuration(inputPath)

Reads the configuration file and dumps it into a Dictionary.
"""
function read_configuration(inputPath::String)
    config = Dict{String, Any}();
    f = open(inputPath);
    for ln in eachline(f)
        if startswith(ln, "#") || length(ln) < 3
            continue;
        end
        name_type, value = split(ln, "=");
        name, ntype = split(strip(name_type), ":");
        value = strip(value);# in case it contains a space at the after the '='
        # available types are: str (default), int, float, bool (lists in the future)
        if ntype == "str"
            value = convert(String, value);
        elseif ntype == "int"
            value = parse(Int64, value);
        elseif ntype == "float"
            value = parse(Float64, value);
        elseif ntype == "bool"
            value = parse(Bool, value);
        elseif ntype == "int_list"
            value = map(x->parse(Int64,x), split(value, ","));
        elseif ntype == "float_list"
            value = map(x->parse(Float64,x), split(value, ","));
        elseif ntype == "str_list"
            value = map(x->strip(x), split(value, ","));
            value = convert(Array{String,1}, value);
        end
        config[name] = value;
    end
    close(f);
    return config;
end

"""
    store_configuration(inputPath, config)

Dumps the configuration dict into a file.
"""
function store_configuration(inputPath::String, config::Dict{String, Any})
    open(inputPath, "w") do f
        for k in keys(config)
            if isa(config[k], String)
                write(f, @sprintf("%s:str=%s\n",k,config[k]));
            elseif isa(config[k], Int64)
                write(f, @sprintf("%s:int=%d\n",k,config[k]));
            elseif  isa(config[k], Float64)
                write(f, @sprintf("%s:float=%f\n",k,config[k]));
            elseif  isa(config[k], Bool)
                if config[k]
                    write(f, @sprintf("%s:bool=true\n",k));
                else
                    write(f, @sprintf("%s:bool=false\n",k));
                end
            elseif  isa(config[k], Array{String,1})
                s = config[k][1];
                for i in collect(2:length(config[k]));
                    s = string(s,",",config[k][i]);
                end                
                write(f, @sprintf("%s:str_list=%s\n",k, s));
            elseif  isa(config[k], Array{Int64,1})
                s = @sprintf("%d",config[k][1]);
                for i in collect(2:length(config[k]))
                    s = string(s,",",@sprintf("%d",config[k][i]) );
                end
                write(f, @sprintf("%s:int_list=%s\n",k, s));
            elseif  isa(config[k], Array{Float64,1})
                s = @sprintf("%f",config[k][1]);
                for i in collect(2:length(config[k]))
                    s = string(s,",",@sprintf("%f",config[k][i]) );
                end                
                write(f, @sprintf("%s:float_list=%s\n",k, s)); 
            end
        end
    end
end
    

function get_dimensions_from_input_file(inputPath::String)
    f = open(inputPath);
    n, d, nnv = map(x->parse(Int64,x), split(readline(f)));
    close(f);
    return n, d
end


"""
    save_data_as_cluto(DATA, path)

This function assumes that the array is in column-order (columns contain the examples)
"""
function sparseMatToFile(D::SparseMatrixCSC{Float64,Int64}, path::String)    
    fout = open(path,"w")

    nnz_vals = nnz(D)
    nrows, ncols = size(D);
    write(fout, @sprintf("%d %d %d\n",ncols, nrows, nnz_vals)); #rememeber: column order
    for c in collect(1:ncols)
        c_indices = D[:,c].nzind;
        c_values = D[:,c].nzval;
        for j in eachindex(c_indices)
            write(fout, @sprintf("%d %0.7f ",c_indices[j], c_values[j]) );
        end
        write(fout, "\n");
    end
    
    close(fout)
end

"""
    sparseMatFromFile(inputPath, [assigned_instances], [objects_as_rows], [l2normalize])

Builds a sparse matrix from the content stored at `inputPath`. 
* Filters the rows indicated in the array `assigned_instances`. If `assigned_instances` is missing all the rows are 
included. 
* The resulting matrix contains the columns of the file in its rows unless `objects_as_rows` is set to true.
* The matrix is built as an exact version of the file, but if `l2normalize` is set, then every object row is normalized.

# Examples
```julia-repl
julia> D = sparseMatFromFile("20newsgroups/20ng_tf_cai.csv")
```
"""
function sparseMatFromFile(inputPath::String; assigned_instances::Array{Int64,1}=Int64[], objects_as_rows::Bool=false, l2normalize::Bool=false)
    """
    sp_from_input_file(inputPath[,instance_array])

    Builds a sparse matrix from the content stored at `inputPath`. Also filters the rows
    indicated in the array `instance_array`. If `instance_array` is missing all the rows are 
    included. The resulting array contains the columns of the file in its rows.
    If `objects_as_rows` (set as false by default) is set to true, then the resulting matrix contains
    objects as rows and features as columns.

    # Examples
    ```julia-repl
    julia> D = get_slice_from_input_file("20newsgroups/20ng_tf_cai.csv")
    ```
    """
    f = open(inputPath)
    n_orig, d, nnv = map(x->parse(Int32, x), split(readline(f)));
    
    if length(assigned_instances) == 0
        assigned_instances = collect(1:n_orig);
    end
    n = length(assigned_instances)
    D = spzeros(Float64, d, n);

    inst_id = 1
    cnt = 0
    for ln in eachline(f)
        if inst_id in assigned_instances
            cnt = cnt + 1;
            instance = split(ln);
            for ix=collect(1:2:length(instance))
                feat_ix = parse(Int64, instance[ix]);
                feat_value = parse(Float64, instance[ix+1]);

                #println(feat_ix,",",inst_id," --> ",feat_value)
                D[feat_ix, cnt] = feat_value;
            end
        end
        inst_id = inst_id + 1;
    end
    close(f)
    
    # normalizing column vectors
    if l2normalize
        for q=collect(1:n)
            norm_v = norm(D[:,q]);
            for qi = D[:,q].nzind
                D[qi,q] = D[qi,q] / norm_v;
            end
        end
    end
    
    if objects_as_rows
        return transpose(D);
    else
        return D
    end
end

function normalize_matrix!(M::SparseMatrixCSC{Float64, Int64})
    n = size(M,2);
    for q=collect(1:n)
        norm_v = norm(M[:,q]);
        for qi in M[:,q].nzind
            M[qi,q] = M[qi,q] / norm_v;
        end
    end
end

function normalize_matrix(M::SparseMatrixCSC{Float64, Int64})
    I = Int64[];
    J = Int64[];
    V = Float64[];
    
    n = size(M,2);
    for q=collect(1:n)
        norm_v = norm(M[:,q]);
        for qi in eachindex(M[:,q].nzind)
            push!(J, q);
            push!(I, M[:,q].nzind[qi]);
            push!(V, M[:,q].nzval[qi] / norm_v);
        end
    end
    nrmM = sparse(I,J,V, size(M,1), size(M,2));
    return nrmM;
end

using HDF5

function store_dense_matrix{vtype}(X::Array{vtype,2}, fname::String; labels::Array{Int8, 1}=Int8[])
    fid=h5open(fname, "w");
    write(fid, "DATA/X", X);
    if length(labels) > 0
        write(fid, "DATA/LABELS", labels);
    end
    close(fid);
    h5writeattr(fname, "DATA", Dict("nrows"=>size(X, 1), "ncols"=>size(X, 2)));
end

function load_dense_matrix(fname::String; with_labels::Bool=false)

    fid=h5open(fname, "r");
        
    if !exists(fid["DATA"], "X")
        throw(ErrorException("No data available"));
    end

    X = read(fid["DATA"]["X"]);
    if exists(attrs(fid["DATA"]), "nrows") && exists(attrs(fid["DATA"]), "ncols")
        nrows = read(attrs(fid["DATA"]), "nrows");
        ncols = read(attrs(fid["DATA"]), "ncols");
        if size(X,1) != nrows
            transpose!(X);
        end
    end
    
    labels = Int8[];
    if with_labels && exists(fid["DATA"], "LABELS")
        labels = read(fid["DATA"]["LABELS"]);
    end

    close(fid);
    if length(labels) > 0
        return X, labels
    end
    return X; 
end

function store_sparse_matrix{vtype}(m::SparseMatrixCSC{vtype,Int64}, fpath::String)
    fid=h5open(fpath, "w");
    I, J, V = findnz(m);
    write(fid, "DATA/I", I);
    write(fid, "DATA/J", J);
    write(fid, "DATA/V", V);
    close(fid);
    h5writeattr(fpath, "DATA", Dict("nrows"=>size(m, 1), "ncols"=>size(m, 2)));
end

function load_sparse_matrix(fname::String)
    fid=h5open(fname, "r");
        
    if !(exists(fid["DATA"], "I") || exists(fid["DATA"], "J")|| exists(fid["DATA"], "V"))
        throw(ErrorException("No data available (X, Y)"));
    end
    I = read(fid["DATA"]["I"]);
    J = read(fid["DATA"]["J"]);
    V = read(fid["DATA"]["V"]);
    spm = sparse(I,J,V);
    close(fid);
    return spm;
end


"""
    load_selected_sparse_matrix(fname, instancess_to_pick)

Load sparse data from HDF5 file but it only chooses the COLUMNS whose indexes are contained
into insts_to_pick.

# Example
```julia-repl
julia> m2 = DSNN_IO.load_selected_sparse_matrix("/tmp/sample_mat.sp", [1,5,10])
```
"""
function load_selected_sparse_matrix(fname::String, cols_to_pick::Array{Int64,1})
    fid=h5open(fname, "r");
        
    if !(exists(fid["DATA"], "I") || exists(fid["DATA"], "J")|| exists(fid["DATA"], "V"))
        throw(ErrorException("No data available (X, Y)"));
    end
    
    indexmap = Dict{Int64, Int64}()
    sort!(cols_to_pick);
    ix = 1;
    for i in cols_to_pick
        #i original index
        #indexmap[i] mapped element
        indexmap[i] = ix;
        ix += 1;
    end
    
    I = read(fid["DATA"]["I"]);
    J = read(fid["DATA"]["J"]);
    V = read(fid["DATA"]["V"]);

    newI = Int64[];
    newJ = Int64[];
    newV = similar(V, 0);#initialized array with 0 element

    for i in eachindex(J)

        if !(J[i] in cols_to_pick)
            continue;
        end
        
        push!(newJ, indexmap[J[i]]);
        push!(newI, I[i]);
        push!(newV, V[i]);
    end

    spm = sparse(newI,newJ,newV, length(unique(I)), length(cols_to_pick));
    close(fid);
    return spm;
end
end
