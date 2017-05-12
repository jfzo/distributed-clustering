
#function execute()
if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end
    
nofworkers = 18
addprocs(nofworkers)
results = Dict{String,Array{Int32,1}}()
    
@everywhere using Distances
@everywhere using StatsBase
@everywhere include("WorkerSNN.jl")
@everywhere include("MasterSNN.jl")
using PyCall
@pyimport clustering_scores as cs #clustering_scores.py must be in the path.
    
A = readdlm("text-data/cure_data.csv", ',', Float64);
DATA = convert(SharedArray, transpose(A[:,1:2]));
N = size(DATA, 2);
    
    
partition = generate_partition(nofworkers, N) #N instances assigned to nofworkers cores.

worker_Eps, worker_MinPts, worker_k, pct_sample = 60, 30, 90, 0.3

real_labels = convert(Array{Int32,1}, A[:,3]);

@time begin
master_work(results, DATA, partition, worker_Eps, worker_MinPts, worker_k, pct_sample, similarity="euclidean");
end
#println("\n*************************************\n",results)
scores = cs.clustering_scores(real_labels[results["sampledpoints"]], results["assignments"], false);

println("Eps:",worker_Eps," MinPts:",worker_MinPts," K:",worker_k," ARI(sk):",
                round(scores["ARI"],5)," VM(sk):",round(scores["VM"],5),
                " #Gen.Samples:",length(results["sampledpoints"]) )


writedlm("./cure_data.csv.results.samples", results["sampledpoints"], ',');
writedlm("./cure_data.csv.results.corepoints", results["corepoints"], ',');    
writedlm("./cure_data.csv.results.assignments", results["assignments"], ',');    

#end


