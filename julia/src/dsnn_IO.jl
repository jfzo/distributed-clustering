module DSNN_IO


"""
    sparseMatFromFile(inputPath)

Builds a sparse matrix from the content stored at _inputPath_. 
* Filters the rows indicated in the array `instance_array`. If `instance_array` is missing all the rows are 
included. 
The resulting array contains the columns of the file in its rows.
If `objects_as_rows` (set as false by default) is set to true, then the resulting matrix contains
objects as rows and features as columns.

# Examples
```julia-repl
julia> D = sparseMatFromFile("20newsgroups/20ng_tf_cai.csv")
```
"""
function sparseMatFromFile(inputFile::String)    
    content = read(inputFile);
    src = IOBuffer(content);
    n = parse(Int64, readuntil(src, " "));
    #println(n)
    d = parse(Int64, readuntil(src, " "));
    #println(d)
    nnv = parse(Int64, readuntil(src, "\n"));

    r = Int64[];
    c = Int64[];
    v = Float64[];


    inst_id = 1
    cnt::Int64 = 0
    for ln in eachline(src)
        cnt = cnt + 1;
        instance = split(ln);

        for ix=collect(1:2:length(instance))

            feat_ix = parse(Int64, instance[ix]);
            feat_value = parse(Float64, instance[ix+1]);

            #println(feat_ix,",",inst_id," --> ",feat_value)
            #D[feat_ix, cnt] = feat_value;
            push!(r, feat_ix);
            push!(c, cnt);
            push!(v, feat_value);
        end

        inst_id = inst_id + 1;
    end
    close(src)
    D = sparse(r,c,v,d,n);
    for q=collect(1:n)
        norm_v = norm(D[:,q]);
        for qi = D[:,q].nzind
            D[qi,q] = D[qi,q] / norm_v;
        end
    end
    return D
end

using HDF5

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
function load_selected_sparse_matrix(fname::String, insts_to_pick::Array{Int64,1})
    fid=h5open(fname, "r");
        
    if !(exists(fid["DATA"], "I") || exists(fid["DATA"], "J")|| exists(fid["DATA"], "V"))
        throw(ErrorException("No data available (X, Y)"));
    end
    I = read(fid["DATA"]["I"]);
    J = read(fid["DATA"]["J"]);
    V = read(fid["DATA"]["V"]);
    newI = Int64[];
    newJ = Int64[];
    newV = similar(V, 0);#initialized array with 0 element
    next_j = 0
    prev_j = 0
    for i in eachindex(J)

        if !(J[i] in insts_to_pick)
            continue;
        end
        if prev_j != J[i]
            next_j += 1
            prev_j = J[i]
        end

        push!(newJ, next_j);
        push!(newI, I[i]);
        push!(newV, V[i]);
    end

    spm = sparse(newI,newJ,newV);
    close(fid);
    return spm;
end
end
