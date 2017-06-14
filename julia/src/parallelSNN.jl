
#function execute()
if length(workers()) > 1
    println("Removing previous workers...")
    rmprocs(workers())
end
    
if length(ARGS) != 4
    println("julia parallelSNN.jl INPUT_MATRIX_FILE LABEL_FILE NrCores output-logfile")
    exit()
end

DATA_PATH = ARGS[1];
LABEL_PATH = ARGS[2];
Ncores = parse(ARGS[3]);
LOGF = ARGS[4];

nofworkers = Ncores
addprocs(nofworkers)
    
push!(LOAD_PATH, pwd())
@everywhere using Distances
@everywhere using StatsBase
@everywhere using Clustering
#@everywhere using WorkerSNN
#@everywhere using MasterSNN
#@everywhere using IOSNN
@everywhere include("WorkerSNN.jl")
@everywhere include("MasterSNN.jl")
@everywhere include("IOSNN.jl")

using PyCall
@pyimport clustering_scores as cs #clustering_scores.py must be in the path.
    


results = Dict{String,Array{Int32,1}}()
println("Opening file:",DATA_PATH," with label file:", LABEL_PATH)

real_labels = readdlm(LABEL_PATH, Int32);
M, dims = get_header_from_input_file(DATA_PATH);

partition = generate_partition(nofworkers, M) #N instances assigned to nofworkers cores.


pct_sample = 0.3;

#range_Eps = [5, 8, 10, 15];
#range_MinPts = collect(5:5:30);
#range_K = [20, 30, 40];

range_Eps = [8,8,8,8,8,8,8,8,8,8];
range_MinPts = [20];
range_K = [30];

max_dsnn_perf = -1;
max_dsnn_perf_tuple = []

logh = open(LOGF,"w");

write(logh, @sprintf("%-10s;%-10s;%-10s;%-10s;%-10s;%-10s;%-10s;%-10s;%-10s;%-10s\n", "ARI", "VM", "Eps", "MinPts","K", "num_clusters", "num_core", "num_sampled", "num_Nnoisy", "num_noisy"));
for Eps=range_Eps
    for MinPts=range_MinPts
        for K=range_K 
            master_work(results, DATA_PATH, partition, Eps, MinPts, K, pct_sample);
            scores = cs.clustering_scores(real_labels[results["sampledpoints"]], results["assignments"], false);
            sampled_pts = length(results["sampledpoints"]);
            
            # write configuration and results
            #ARI, VM, NUMCLUSTERS, NUMCOREPTS, SAMPLEDPts,NUMNONNOISYPTS, NUMNOISYPTS
            num_clusters = length(results["labels"]);
            num_core = length(results["corepoints"]);
            num_sampled = length(results["sampledpoints"]);
            num_nnoisy = length(find(x->x>0, results["assignments"]));
            num_noisy = length(find(x->x==0, results["assignments"]));
            write(logh, @sprintf("%0.4f; %0.4f;   %d; %d; %d;    %d; %d; %d; %d; %d\n", scores["ARI"], scores["VM"], Eps, MinPts, K, num_clusters, num_core, num_sampled, num_nnoisy, num_noisy));
            # 
        end
        flush(logh)
    end
end

close(logh)

#end


