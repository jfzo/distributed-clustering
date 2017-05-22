
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
            
            println(feat_ix,",",inst_id," --> ",feat_value)
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
