
function save_data_as_cluto(D::Array{Float64,2}, path::String; with_labels=false, label_file="")
    # This function assumes that the array is in column-order (columns contain the examples)
    # with_labels denotes that the matrix hax the label information in the last row 
    fout = open(path,"w")
    d, n = size(D) # Rows contain features and columns the examples.
    if with_labels
        d = d - 1
    end
    
    nnz = 0;
    write(fout, @sprintf("                         \n"));
    for row = collect(1:n)
        for col = collect(1:d)
            if D[col, row] != 0
                write(fout, @sprintf("%d %0.5f ",col, D[col, row]) );
                nnz += 1;
            end
        end
        write(fout, "\n");
    end
    
    seekstart(fout)
    write(fout, @sprintf("%d %d %d",n,d,nnz))
    
    close(fout)
    
    if with_labels && length(label_file) > 0
        lblout = open(label_file, "w")
        for example = collect(1:n)
            write(lblout, @sprintf("%d\n", D[end, example]) )
        end
        close(lblout)
    end
end

function save_data_as_cluto(D::SparseMatrixCSC{Float64,Int64}, path::String; with_labels=false, labels::Array{Float64,2}=[], label_file="")
    # This function assumes that the array is in column-order (columns contain the examples)
    # with_labels denotes that the matrix hax the label information in the last row 
    fout = open(path,"w")
    d, n = size(D) # Rows contain features and columns the examples.
    
    nnz = 0;
    write(fout, @sprintf("                         \n"));
    for row = collect(1:n)
        for col = collect(1:d)
            if D[col, row] != 0
                write(fout, @sprintf("%d %0.5f ",col, D[col, row]) );
                nnz += 1;
            end
        end
        write(fout, "\n");
    end
    
    seekstart(fout)
    write(fout, @sprintf("%d %d %d",n,d,nnz))
    
    close(fout)
    
    if with_labels && length(label_file) > 0
        lblout = open(label_file, "w")
        for example = collect(1:n)
            write(lblout, @sprintf("%d\n", labels[example]) )
        end
        close(lblout)
    end
end

function get_cluto_data(D::Array{Float64,2}, path::String)
    f = open(path)
    n, d, nnv = map(x->parse(Int64,x), split(readline(f)));

    #D = SharedArray(Float64, (d,n))

    inst_id = 1
    for ln in eachline(f)
        instance = split(ln);
        for ix=collect(1:2:length(instance))
            feat_ix = parse(Int64, instance[ix]);
            feat_value = parse(Float64, instance[ix+1]);
            
            #println(feat_ix,",",inst_id," --> ",feat_value)
            D[feat_ix, inst_id] = feat_value;
        end
        inst_id = inst_id + 1;
    end

    close(f)
end


function get_header_from_input_file(inputPath::String)
    f = open(inputPath)
    n, d, nnv = map(x->parse(Int64,x), split(readline(f)));
    close(f)
    return n, d
end

function get_slice_from_input_file(D::Array{Float64,2}, inputPath::String, assigned_instances::Array{Int64,1})
    f = open(inputPath)
    n, d, nnv = map(x->parse(Int64,x), split(readline(f)));

    assert(size(D,2) == length(assigned_instances))
    #D = Array(Float64, (d, length(assigned_instances)))

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
    return D
end
