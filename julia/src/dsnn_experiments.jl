#module DSNNExperiments
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--inputfile", "-i"
            help = "an option with an argument"
            required = true        
        "--labelfile", "-l"
            required = true
        "--nworkers", "-w"
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
nofworkers=parse(parsed_args["nworkers"])



#function perform_experiments(DATA_PATH, LABEL_PATH, nofworkers)
if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
else
    println("No worker to remove!")
end

#addprocs(nofworkers)
addprocs(["root@158.251.93.5:3302","root@158.251.93.5:3303","root@158.251.93.5:3304","root@158.251.93.5:3305","root@158.251.93.5:3306","root@158.251.93.5:3307","root@158.251.93.5:3308","root@158.251.93.5:3309","root@158.251.93.5:3310","root@158.251.93.5:3311","root@158.251.93.5:3312","root@158.251.93.5:3313","root@158.251.93.5:3314","root@158.251.93.5:3315"])
push!(LOAD_PATH, pwd())

@everywhere using Distances
@everywhere using StatsBase
#@everywhere import Clustering 
@everywhere using LightGraphs
include("MasterSNN.jl")
@everywhere include("WorkerSNN.jl")
@everywhere include("IOSNN.jl")
@everywhere include("SNNDBSCAN.jl")
@everywhere include("SNNGraphUtil.jl")

using PyCall
@pyimport clustering_scores as cs #clustering_scores.py must be in the path.
using JLD

#real_labels = vec(readdlm(LABEL_PATH, Int32));
N, dim = get_header_from_input_file(DATA_PATH);
real_labels = zeros(N,1)

#DATA = zeros(dim,N);
#get_cluto_data(DATA, DATA_PATH);
println("Dataset ",DATA_PATH," (#Docs:",N,"/#Features:",dim,") and Num. of workers:",nofworkers);
pct_sample = 3; pct_sample = pct_sample/100; # (%) percentage of each local worker that will be sampled and transmitted to the Mas2ter

#global score statistics (along cut_point values)
nruns = 5;# number of runs per cut_point value
#cut_values = collect(50:10:150);
cut_values = [50];

k_range=[50]
worker_eps_start_val, worker_eps_step_val, worker_eps_end_val=10.0, 10.0, Inf;
worker_minpts_start_val, worker_minpts_step_val, worker_minpts_end_val=10, 10, Inf


summary_scores = Dict{String, Any}("elapsed"=>[], "bytesalloc" => [], "E"=>[], "P" => [], "ARI" => [], "AMI" => [], "NMI" => [], "H" => [], "C" => [], "VM" => [], "cut_range" => cut_values, "nruns" => nruns, "nworkers" => nofworkers, "worker_epsilon_range"=>(worker_eps_start_val, worker_eps_step_val, worker_eps_end_val), "worker_minpts_range"=>(worker_minpts_start_val, worker_minpts_step_val, worker_minpts_end_val), "worker_k_range"=>k_range)

for cut_point=cut_values
    @printf "Starting runs with snn_cut_point:%d\n" cut_point 
    #score values attained along runs
    run_scores = Dict{String, Array{Float64, 1}}("elapsed"=>[], "bytesalloc" => [], "E"=>[], "P" => [], "ARI" => [], "AMI" => [], "NMI" => [], "H" => [], "C" => [], "VM" => [])

    for run_no=collect(1:nruns)
        partition = generate_partition(nofworkers, N); #N instances assigned to nofworkers cores.
        # Performs the clustering task
        results = Dict{String,Any}()        

        _, elapsed_t, bytes_alloc, _, _ = @timed master_work(results, DATA_PATH, partition, pct_sample, 
        k_range, worker_eps_start_val, worker_eps_step_val, worker_eps_end_val, worker_minpts_start_val, worker_minpts_step_val, worker_minpts_end_val, similarity="cosine", KNN=7, snn_cut_point=cut_point);

        push!(run_scores["elapsed"], elapsed_t)
        push!(run_scores["bytesalloc"], bytes_alloc)

        scores = cs.clustering_scores(real_labels, results["assignments"], false);
        scores = convert(Dict{String, Float64}, scores);
        for qm=keys(scores)
            push!(run_scores[qm], scores[qm])
            end    
    end
    #compute mean and std for each score (along runs)
    for qm=keys(run_scores)
        qm_mean = mean(run_scores[qm])
        qm_std = std(run_scores[qm])
        push!(summary_scores[qm], (qm_mean, qm_std))
        end 
end
# save summary_scores
jldoutput = join([DATA_PATH[1:end-4],"_summary.jld"]);#assumes that the data file ends with '.csv'
JLD.save(jldoutput, "summary_scores", summary_scores)
println("Storing summary to:", jldoutput)

#end
