using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--config", "-c"
            help = "an option with an argument"
            required = true        
    end

    return parse_args(s)
end



parsed_args = parse_commandline()

CONFIG_FILE=parsed_args["config"]


if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end


include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_Experiment.jl")
# Grancoloso
#addprocs(["158.251.93.5:3308","158.251.93.5:3307","158.251.93.5:3306","158.251.93.5:3305",])
#addprocs(["158.251.93.5:3308","158.251.93.5:3307","158.251.93.5:3306","158.251.93.5:3305",
#        "158.251.93.5:3304","158.251.93.5:3303","158.251.93.5:3302","158.251.93.5:3301",])

# Coloso
#addprocs(["158.251.88.180:3301","158.251.88.180:3302","158.251.88.180:3303","158.251.88.180:3304",])

config = DSNN_IO.read_configuration(CONFIG_FILE);
addprocs(config["master.nodelist"]);

@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_KNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_Master.jl")

using Graphs
using CSV

results = Dict{String,Any}();

run_seed = config["seed"];
srand(run_seed);
DATA_PATH = config["master.inputpath"];
BENCHMARK = config["benchmark"];
DATA_LEN, DATA_DIM = DSNN_IO.get_dimensions_from_input_file(DATA_PATH);
partitions = DSNN_Master.generate_partitions(length(workers()), DATA_LEN); # N must be extracted from the data.


println("\n\n***********************************************************")
println(DSNN_EXPERIMENT.config_as_str(config));
DSNN_Master.start(results, DATA_PATH, partitions, config);
#storing final result
writedlm(@sprintf("%s.dsnnfinal.labels",DATA_PATH), results["stage2_labels"], "\n");


# Experimentation over the obtained corepoints
D = DSNN_IO.sparseMatFromFile(DATA_PATH, l2normalize=true);
real_labels = vec(readdlm(@sprintf("%s.labels",DATA_PATH), Int32));

Dw = D[:,results["stage1_corepoints"]];
cp_real_labels = real_labels[results["stage1_corepoints"]];

CSV.write(@sprintf("%s.corepoints.csv",DATA_PATH), DataFrames.DataFrame(full(transpose(Dw))), delim=' ',  header=false);
writedlm(@sprintf("%s.corepoints.labels",DATA_PATH), cp_real_labels, "\n");
println("Corepoint labels identified in Stage-2 were stored in file ",@sprintf("%s.corepoints.labels",DATA_PATH));


snnmat, knnmat = DSNN_KNN.get_snnsimilarity(Dw, config["master.stage2knn"], l2knng_path=config["l2knng.path"]);

adj_mat = snnmat;
if config["master.use_snngraph"]
    snngraph = DSNN_KNN.get_snngraph(knnmat, snnmat);
    adj_mat = snngraph;
end

@time begin
    DSNN_EXPERIMENT.perform_corepoint_snn(adj_mat, config);
end

@time begin
    DSNN_EXPERIMENT.perform_corepoint_conncomps(adj_mat, config);
end

@time begin
    DSNN_EXPERIMENT.perform_corepoint_maxcliques(adj_mat, config);
end

@time begin
    DSNN_EXPERIMENT.perform_corepoint_lblprop(adj_mat, config);
end

@time begin
    DSNN_EXPERIMENT.perform_corepoint_dbscan(adj_mat, config);
end

run(`python evaluate_corepoint_files.py -e snn,dbscan,conncomps,maxcliques,lblprop -i $DATA_PATH -b $BENCHMARK -f rst`);
