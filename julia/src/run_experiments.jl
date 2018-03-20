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

config = DSNN_IO.read_configuration(CONFIG_FILE);
addprocs(config["master.nodelist"]);

output = open(config["logging.path"], "w");
println("Logging file: ", config["logging.path"]);

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


#println("\n\n***********************************************************")
write(output, "\n***********************************************************\n");
#println(DSNN_EXPERIMENT.config_as_str(config));
write(output, DSNN_EXPERIMENT.config_as_str(config));
write(output,"\n");
tic();
DSNN_Master.start(results, DATA_PATH, partitions, config);
elapsed_t = toc();
#storing final result
write(output, "***** TIMERS ******\n");
write(output, @sprintf("elapsed_worker_clustering: %f\n", results["elapsed_worker_clustering"]));
write(output, @sprintf("elapsed_master_wresults_join: %f\n", results["elapsed_master_wresults_join"]));
write(output, @sprintf("elapsed_master_similarity_computation: %f\n", results["elapsed_master_similarity_computation"]));
write(output, @sprintf("elapsed_master_clustering: %f\n", results["elapsed_master_clustering"]));
write(output, @sprintf("elapsed_stage2_retransmission: %f\n", results["elapsed_stage2_retransmission"]));
write(output, @sprintf("elapsed_master_final_label_join: %f\n", results["elapsed_master_final_label_join"]));
write(output, @sprintf("Elapsed time: %f\n", elapsed_t));
writedlm(@sprintf("%s.dsnnfinal.labels",DATA_PATH), results["stage2_labels"], "\n");
write(output, "Final labels stored at ",@sprintf("%s.dsnnfinal.labels\n",DATA_PATH));



# Experimentation over the obtained corepoints
D = DSNN_IO.sparseMatFromFile(DATA_PATH, l2normalize=true);
real_labels = vec(readdlm(@sprintf("%s.labels",DATA_PATH), Int32));

Dw = D[:,results["stage1_corepoints"]];
cp_real_labels = real_labels[results["stage1_corepoints"]];

CSV.write(@sprintf("%s.corepoints.csv",DATA_PATH), DataFrames.DataFrame(full(transpose(Dw))), delim=' ',  header=false);
writedlm(@sprintf("%s.corepoints.labels",DATA_PATH), cp_real_labels, "\n");
#println("Corepoint labels identified in Stage-2 were stored in file ",@sprintf("%s.corepoints.labels",DATA_PATH));
write(output, "Corepoint labels identified in Stage-1 were stored in file ",@sprintf("%s.corepoints.labels (%d)\n",DATA_PATH, length(results["stage1_corepoints"])));


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

#run(`python evaluate_corepoint_files.py -e snn,dbscan,conncomps,maxcliques,lblprop -i $DATA_PATH -b $BENCHMARK -f rst`);
s_performance = readstring(`python evaluate_corepoint_files.py -e snn,dbscan,conncomps,maxcliques,lblprop -i $DATA_PATH -b $BENCHMARK -f rst`);
write(output, s_performance);
write(output, "\n");
close(output);
