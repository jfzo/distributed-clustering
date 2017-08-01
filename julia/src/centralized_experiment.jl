#=
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--inputfile", "-i"
            help = "an option with an argument"
            required = true        
        "--labelfile", "-l"
            required = true
    end

    return parse_args(s)
end



parsed_args = parse_commandline()


#DATA_PATH = "./20newsgroups/20ng_tfidf_cai_top10.csv";
#LABEL_PATH = "./20newsgroups/20ng_tfidf_cai_top10.csv.labels";
#DATA_PATH = "./cure_large.dat";
#LABEL_PATH = "./cure_large.dat.labels";
DATA_PATH=parsed_args["inputfile"]
LABEL_PATH=parsed_args["labelfile"]
=#

using StatsBase
using Distances
include("IOSNN.jl");
include("WorkerSNN.jl");
include("SNNDBSCAN.jl");
using Clustering
using PyCall
using ArgParse
@pyimport clustering_scores as cs #clustering_scores.py must be in the path.

function snn_grid_evaluation(DATA_PATH::String, LABEL_PATH::String)


    
    println(@sprintf("Opening file %s with labels in %s", DATA_PATH, LABEL_PATH))
    real_labels = vec(readdlm(LABEL_PATH, Int32));
    num_points, dim = get_header_from_input_file(DATA_PATH);

    D = zeros(dim, num_points);
    get_cluto_data(D, DATA_PATH);

    S = zeros(Float64, num_points, num_points);
    cosine_sim(D, S);
    # Empty matrix that will be filled iteratively for each value of K
    Snn = zeros(Float64, num_points, num_points);

    #=
    range_Eps = [5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80];
    range_MinPts = [5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80];
    range_K = [10 20 30 40 50 70 90 110];
    =#


    k_range = collect(30:10:200)
    max_perf = -1;
    max_perf_tuple = []

    for K=k_range
        shared_nn_sim(D, K, Snn, S)

        Eps_range = collect(5.0:10.0:K)
        MinPts_range = collect(5:10:K)

        for Eps=Eps_range
            for MinPts=MinPts_range

                d_point_cluster_id = Dict{Int64, Int64}();            
                cluster_assignment = fill(SNNDBSCAN.UNCLASSIFIED, num_points);
                corepoints = Int64[];
                for i=collect(1:num_points)
                    d_point_cluster_id[i] = SNNDBSCAN.UNCLASSIFIED;
                end

                _, elapsed_t, _, _, _ = @timed SNNDBSCAN.dbscan(num_points, Eps, MinPts, Snn, d_point_cluster_id, corepoints)

                for i=collect(1:num_points)
                    cluster_assignment[i] = d_point_cluster_id[i]
                end 

                scores = cs.clustering_scores(real_labels, cluster_assignment, false);
                if scores["VM"] > max_perf
                    max_perf = scores["VM"];
                    max_perf_tuple = (Eps, MinPts, K, elapsed_t);
                    #println("[Current best ARI] Eps:",Eps," MinPts:",MinPts," K:",K," ARI:",scores["ARI"]," VM:",scores["VM"])
                end

            end
        end
    end
    
    return Dict{String, Real}("elapsed"=>max_perf_tuple[4], "epsilon" => max_perf_tuple[1], "minpts"=>max_perf_tuple[2], "K" => max_perf_tuple[3], "VM" => max_perf)
    #return (max_perf, max_perf_tuple[1],max_perf_tuple[2],max_perf_tuple[3],max_perf_tuple[4])

end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--inputfile", "-i"
            help = "an option with an argument"
            required = true        
        "--labelfile", "-l"
            required = true
    end

    return parse_args(s)
end


if ~isinteractive()

    parsed_args = parse_commandline()

    DATA_PATH=parsed_args["inputfile"]
    LABEL_PATH=parsed_args["labelfile"]

    perf = snn_grid_evaluation(DATA_PATH, LABEL_PATH)
    println( @sprintf("\nBest VM score:%0.4f attained with parameters epsilon:%0.0f minpts:%d K:%d Time:%0.4f", perf["VM"], perf["epsilon"],perf["minpts"],perf["K"],perf["elapsed"]) )

end   

