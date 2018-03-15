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


function execute_run()
                write(output, "\n***********************************************************\n");
                write(output, DSNN_EXPERIMENT.config_as_str(config));
                write(output,"\n");
                tic();
                DSNN_Master.start(results, DATA_PATH, partitions, config);
                elapsed_t = toc();
                if length(results["stage1_corepoints"]) < 3#at least two clusters
                    write(output, @sprintf("Not enough corepoints (%d). Aborting execution of evaluation\n", length(results["stage1_corepoints"])) );
                    return -1;
                end
                #storing final result
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
                write(output, "Corepoint labels identified in Stage-1 were stored in file ",@sprintf("%s.corepoints.labels (%d)\n",DATA_PATH, length(results["stage1_corepoints"])));


                snnmat, knnmat = DSNN_KNN.get_snnsimilarity(Dw, config["master.stage2knn"], l2knng_path=config["l2knng.path"]);

                adj_mat = snnmat;
                if config["master.use_snngraph"]
                    snngraph = DSNN_KNN.get_snngraph(knnmat, snnmat);
                    adj_mat = snngraph;
                end

                DSNN_EXPERIMENT.perform_corepoint_snn(adj_mat, config);
                DSNN_EXPERIMENT.perform_corepoint_conncomps(adj_mat, config);
                DSNN_EXPERIMENT.perform_corepoint_dbscan(adj_mat, config);

                s_performance = readstring(`python evaluate_corepoint_files.py -e snn,dbscan,conncomps -i $DATA_PATH -b $BENCHMARK_LABELS -f rst`);
                write(output, s_performance);
                write(output, "\n");
end


parsed_args = parse_commandline()

CONFIG_FILE=parsed_args["config"]


if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end


include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
#@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_KNN.jl")
#@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
#@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_Master.jl")


config = DSNN_IO.read_configuration(CONFIG_FILE);
addprocs(config["master.nodelist"]);

@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_IO.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_KNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_SNN.jl")
@everywhere include("/workspace/distributed_clustering/julia/src/dsnn_Master.jl")
include("/workspace/distributed_clustering/julia/src/dsnn_Experiment.jl")

using Graphs
using CSV

#config = DSNN_IO.read_configuration(CONFIG_FILE);
#DSNN_IO.store_configuration(CONFIG_FILE, config)

"""
                ^^ Parameters to tune ^^
  ** Parameter **               ** recommended value (empirically)**     ** tested values ** 
master.stage2snnsim_threshold              0.9                            [0.0, 0.5, 0.9]
master.stage2knn                           70                             [30, 70, 120]

master.stage2clustering                    snn (otherwise the two params below are useless)
master.snn.minpts                          3                              [3, 8, 13, 20]
master.snn.eps                             0.8                            [0.1, 0.5, 0.7, 0.9]

worker.knn                                 70                             [70, 120]
worker.snn_minpts                          13                             [3, 8, 13, 20]
worker.snn_eps                             0.9                            [0.1, 0.5, 0.7, 0.9]
"""


results = Dict{String,Any}();
run_seed = config["seed"];
BENCHMARK_LABELS = config["benchmark"];
srand(run_seed);
DATA_PATH = config["master.inputpath"];
DATA_LEN, DATA_DIM = DSNN_IO.get_dimensions_from_input_file(DATA_PATH);
partitions = DSNN_Master.generate_partitions(length(workers()), DATA_LEN); # N must be extracted from the data.

output = open(config["logging.path"], "w");

config["master.stage2clustering"] = "snn"
config["master.use_snngraph"] = false
for master_stage2snnsim_threshold in [0.0, 1e-7, 1e-6, 1e-5]
    for master_stage2knn in [50, 120]
        for master_snn_minpts in [3, 5, 8, 13, 20]
            for master_snn_eps in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1]
                config["master.stage2snnsim_threshold"] = master_stage2snnsim_threshold;
                config["master.stage2knn"] = master_stage2knn;
                config["master.snn.minpts"] = master_snn_minpts;
                config["master.snn.eps"] = master_snn_eps;
                println();

                println(DSNN_EXPERIMENT.config_as_str(config));

                try
                    execute_run();
                    catch y
                    if isa(y, ErrorException)
                        println(y)
                        write(output, "Error occurred!");
                    end
                end

            end
        end
    end
end


#=
config["master.stage2clustering"] = "conncomps"
#for master_snn_minpts in [3, 5, 8, 13, 20]
#    for master_snn_eps in [0.001, 0.01, 0.03, 0.05, 0.1]
        for master_stage2snnsim_threshold in [0.0]
            for master_stage2knn in [3]
                for worker_knn in [120]
                    for worker_snn_minpts in [5]
                        for worker_snn_eps in [1e-7, 1e-6, 5e-6]
                            config["master.stage2snnsim_threshold"] = master_stage2snnsim_threshold;
                            config["master.stage2knn"] = master_stage2knn;                    
                            #config["master.snn.minpts"] = master_snn_minpts;
                            #config["master.snn.eps"] = master_snn_eps;                            
                            config["worker.knn"] = worker_knn;
                            config["worker.snn_minpts"] = worker_snn_minpts;
                            config["worker.snn_eps"] = worker_snn_eps;

                            println();
                            println(DSNN_EXPERIMENT.config_as_str(config));

                            try
                                execute_run();
                                catch y
                                if isa(y, ErrorException)
                                    write(output, y);
                                end
                            end
                            
                        end
                    end
                end               
            end
        end
#    end
#end
=#

close(output);
