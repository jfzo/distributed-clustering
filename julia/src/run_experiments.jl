using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--inputfile", "-i"
            help = "an option with an argument"
            required = true        
    end

    return parse_args(s)
end



parsed_args = parse_commandline()

DATA_PATH=parsed_args["inputfile"]


if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end


include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
# Grancoloso
#addprocs(["158.251.93.5:3308","158.251.93.5:3307","158.251.93.5:3306","158.251.93.5:3305",])
#addprocs(["158.251.93.5:3308","158.251.93.5:3307","158.251.93.5:3306","158.251.93.5:3305",
#        "158.251.93.5:3304","158.251.93.5:3303","158.251.93.5:3302","158.251.93.5:3301",])

# Coloso
#addprocs(["158.251.88.180:3301","158.251.88.180:3302","158.251.88.180:3303","158.251.88.180:3304",])

overall_parameters = DSNN_IO.read_configuration("experiments_config.csv");
addprocs(overall_parameters["master.nodelist"]);

@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_KNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_Master.jl")

using Graphs
using CSV

results = Dict{String,Any}();

run_seed = overall_parameters["seed"];
srand(run_seed);

DATA_LEN, DATA_DIM = DSNN_IO.get_dimensions_from_input_file(DATA_PATH);
partitions = DSNN_Master.generate_partitions(length(workers()), DATA_LEN); # N must be extracted from the data.


println("\n\n***********************************************************")
DSNN_Master.start(results, DATA_PATH, partitions, overall_parameters);
#storing final result
writedlm(@sprintf("%s.dsnnfinal.labels",DATA_PATH), results["stage2_labels"], "\n");


# Experimentation over the obtained corepoints
D = DSNN_IO.sparseMatFromFile(DATA_PATH, l2normalize=true);
real_labels = vec(readdlm(@sprintf("%s.labels",DATA_PATH), Int32));

Dw = D[:,results["stage1_corepoints"]];
cp_real_labels = real_labels[results["stage1_corepoints"]];

CSV.write(@sprintf("%s.corepoints.csv",DATA_PATH), DataFrames.DataFrame(full(transpose(Dw))), delim=' ',  header=false);
writedlm(@sprintf("%s.corepoints.labels",DATA_PATH), cp_real_labels, "\n");

snnmat, knnmat = DSNN_KNN.get_snnsimilarity(Dw, overall_parameters["master.stage2knn"], l2knng_path=overall_parameters["l2knng.path"]);

adj_mat = snnmat;
if overall_parameters["master.use_snngraph"]
    snngraph = DSNN_KNN.get_snngraph(knnmat, snnmat);
    adj_mat = snngraph;
end

# Applying SNN-Clustering overthe corepoints
@time begin
    println("Applying SNN-Clustering over the corepoints...");
    cp_results = DSNN_SNN.snn_clustering(overall_parameters["master.snn.eps"], overall_parameters["master.snn.minpts"], adj_mat);

    labels_found = fill(0, size(cp_results["labels"],1));
    for c in collect(1:size(cp_results["labels"],2))
        for i in cp_results["labels"][:,c].nzind
            labels_found[i] = cp_results["clusters"][c]; # extract the right assigned label name
        end
    end

    println("Num. Clusters found:",length(unique(labels_found)))
    if length(find(x->x<0, cp_results["clusters"])) > 0
        println(@sprintf("Percentage of noise:: %0.2f", 
                length(cp_results["labels"][:,1].nzind)/size(cp_results["labels"],1)))
    else
        println("Amount of noise: 0");
    end
    writedlm(@sprintf("%s.corepoints.snn.labels",DATA_PATH), labels_found, "\n");
end

# Applying Connected components
@time begin
    println("Applying Connected Components over the corepoints...");
    numpoints = size(Dw,2);
    println("Num. points:",numpoints);

    G = Graphs.simple_adjlist(numpoints, is_directed=false);
    for i in collect(1:numpoints)
        for j in adj_mat[:,i].nzind
            Graphs.add_edge!(G, i, j)
        end
    end

    cmps = Graphs.connected_components(G);

    println("Num. connected components:",length(cmps));
    labels_found = fill(-1, numpoints);
    for cmp_i in eachindex(cmps)
        for p in cmps[cmp_i]
            labels_found[p] = cmp_i;
        end
    end
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.conncomps.labels",DATA_PATH), labels_found, "\n");

end

# Applying Maximal Clique to the corepoints
@time begin
    println("Applying Maximal Clique over the corepoints...");
    numpoints = size(Dw,2);
    println("Num. points:",numpoints);

    G = Graphs.simple_adjlist(numpoints, is_directed=false);
    for i in collect(1:numpoints)
        for j in adj_mat[:,i].nzind
            Graphs.add_edge!(G, i, j)
        end
    end

    cmps = Graphs.maximal_cliques(G);

    println("Num. Cliques:",length(cmps));
    labels_found = fill(-1, numpoints);
    for cmp_i in eachindex(cmps)
        for p in cmps[cmp_i]
            labels_found[p] = cmp_i;
        end
    end

    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.cliques.labels",DATA_PATH), labels_found, "\n");

end

# Applying Label propagation to the corepoints
@time begin
    println("Applying Label propagation over the corepoints...")
    using LightGraphs

    G = LightGraphs.Graph(numpoints)
    for i in collect(1:numpoints)
       for j in adj_mat[:, i].nzind
           if j > i 
               # maybe a threshold based on adj_mat[j,i] could be used !
               if ~LightGraphs.add_edge!(G, i, j)
                   println("[M] Error: Cannot add edge between vertices ",i," and ",j)
               end
           end
       end
    end
    labels_found, conv_history = LightGraphs.label_propagation(G);
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.lblprop.labels",DATA_PATH), labels_found, "\n");

end


# Applying DBSCAN over the corepoints
@time begin
    println("Applying DBSCAN over the corepoints...");
    using Clustering

    #dbscan_cl = Clustering.dbscan(full(Dw), 0.1, min_neighbors=15);
    dbscan_cl = Clustering.dbscan(full(1.0 .- adj_mat), overall_parameters["master.dbscan.eps"], overall_parameters["master.dbscan.minpts"]);
    labels_found = dbscan_cl.assignments;
    println("Num. Clusters found:",length(unique(labels_found)))
    writedlm(@sprintf("%s.corepoints.dbscan.labels",DATA_PATH), labels_found, "\n");
end 

