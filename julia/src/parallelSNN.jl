
#function execute()
if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end
    
nofworkers = 12
addprocs(nofworkers)
results = Dict{String,Array{Int32,1}}()
    
@everywhere using Distances
@everywhere using StatsBase
@everywhere include("WorkerSNN.jl")
@everywhere include("MasterSNN.jl")
@everywhere include("IO.jl")
using PyCall
@pyimport clustering_scores as cs #clustering_scores.py must be in the path.
    
DATA_PATH = "text-data/AP_out.dat";
LABEL_PATH = "text-data/AP_out.dat.labels";

real_labels = readdlm(LABEL_PATH, Int32);
#DATA = convert(SharedArray, DATA);
N, dims = get_header_from_input_file(DATA_PATH);
#get_slice_from_input_file(inputPath::String, assigned_instances::Array{Int64,1})

#N = size(DATA, 2);    
partition = generate_partition(nofworkers, N) #N instances assigned to nofworkers cores.

#worker_Eps, worker_MinPts, worker_k, pct_sample = 8, 25, 30, 0.3

#=
range_Eps = [20 30 40 50 60 70 80];
range_MinPts = [20 30 40 50 60 70 80];
range_K = [30 50 70 90 110];
=#
pct_sample = 0.3;

range_Eps = vcat([3, 5, 8, 10], 15:5:50);
range_MinPts = collect(5:5:30);
range_K = [50, 70, 90, 110];

max_dsnn_perf = -1;
max_dsnn_perf_tuple = []

for Eps=range_Eps
    for MinPts=range_MinPts
        for K=range_K 
            #@time begin
                #master_work(results, DATA_PATH, partition, Eps, MinPts, K, pct_sample);
            #end
            
            master_work(results, DATA_PATH, partition, Eps, MinPts, K, pct_sample);
            
            scores = cs.clustering_scores(real_labels[results["sampledpoints"]], results["assignments"], false);
            sampled_pts = length(results["sampledpoints"]);

            # contraint over the nr of sampled points.
            if sampled_pts >= (0.1*N)
                if scores["VM"] > max_dsnn_perf ||  (scores["VM"] > (max_dsnn_perf - 0.2) && sampled_pts > max_dsnn_perf_tuple[4])
                    max_dsnn_perf = scores["VM"];
                    max_dsnn_perf_tuple = (Eps, MinPts, K, sampled_pts);
                    println("[Current best VM] Eps:",Eps," MinPts:",MinPts," K:",K," ARI(sk):",
                    round(scores["ARI"],5)," VM(sk):",round(scores["VM"],5)," #Gen.Samples:",length(results["sampledpoints"]))
                end
            end
        end
    end
end


#=
println("Eps:",worker_Eps," MinPts:",worker_MinPts," K:",worker_k," ARI(sk):",
                round(scores["ARI"],5)," VM(sk):",round(scores["VM"],5),
                " #Gen.Samples:",length(results["sampledpoints"]) )


writedlm("./20ng.results.samples", results["sampledpoints"], ',');
writedlm("./20ng.results.corepoints", results["corepoints"], ',');    
writedlm("./20ng.results.assignments", results["assignments"], ',');    
=#
#end


